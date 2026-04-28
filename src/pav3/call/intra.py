"""Intra-alignment variant calling.

Intra-alignment variants are contained in single alignment records. SNV and INS/DEL variants are identified from
alignment operations (encoded in CIGAR string in SAM/BAM, extracted to a list of operations in PAV).

INV variants are identified by searching for signatures of aberrant alignments that occur when a sequence is aligned
through an inversion without splitting it into multiple records. In this case, matching INS/DEL variants (close
proximity and similar length) and clusters of SNVs and indels are often found near the center of the inversion. These
signatures are identified and tested for an inversion using a kernel density estimate (KDE) of forward and reverse
k-mers between the reference and query in that region. This rarely identifies inversions since most do cause the
alignment to split into multiple records at the alignment, which is left to inter-alignment variant calling implemented
in a separate module in PAV (see :mod:`pav3.lgsv`).
"""

__all__ = [
    'CALL_SOURCE',
    'DEFAULT_BATCH_SIZE',
    'variant_tables_snv_insdel',
    'variant_tables_inv',
    'variant_flag_inv'
]

import agglovar
from pathlib import Path
from typing import Optional

import Bio.Seq
import Bio.SeqIO
import numpy as np
import polars as pl

from .. import schema

from ..align import op

from ..align.lift import AlignLift
from ..align.score import get_score_model
from ..inv import cluster_table, get_inv_row, try_intra_region
from ..io import NullContext, TempDirContainer
from ..kde import KdeTruncNorm
from ..region import Region
from ..params import PavParams
from ..seq import LRUSequenceCache

from . import expr
from .util import COMPL_TR_FROM, COMPL_TR_TO


# Tag variants called with this source
CALL_SOURCE: str = 'INTRA'
"""Variant call source column value."""

DEFAULT_BATCH_SIZE: int = 5_000_000
"""Default batch size for variant table accumulation."""


def variant_tables_snv_insdel(
        df_align: pl.DataFrame | pl.LazyFrame,
        ref_fa_filename: str,
        qry_fa_filename: str,
        batch: int | bool = False,
        sink_snv: Optional[Path | str] = None,
        sink_insdel: Optional[Path | str] = None,
        temp_dir_name: Optional[str] = None,
        pav_params: Optional[PavParams] = None,
) -> Optional[tuple[pl.DataFrame, pl.DataFrame]]:
    """Call variants from alignment operations.

    Calls variants in two separate tables, SNVs in the first, and INS/DEL (including indel and SV) in the second.

    Each chromosome is processed separately. Per-chromosome results are sorted and either held in memory or written
    to a temporary parquet file (when `batch` is set). The final tables concatenate per-chromosome results in
    chromosome sort order.

    If `sink_snv` and `sink_insdel` are provided, the function writes output to those paths and returns None.
    This is required when `batch` is set, because temporary files expire when the function returns.

    Variant sort order within each chromosome: position (pos), alternate base (alt, SNVs) or end position
    (end, non-SNV), alignment score (highest first), query ID (qry_id), and query position (qry_pos).

    :param df_align: Assembly alignments before alignment trimming.
    :param ref_fa_filename: Reference FASTA file name.
    :param qry_fa_filename: Assembly FASTA file name.
    :param batch: If True, use the default batch size and write per-chromosome results to temporary parquet files
        to reduce peak memory. If an integer greater than 0, use that as the batch size threshold. If False or 0,
        hold all results in memory. `sink_snv` and `sink_insdel` are required when batch is enabled.
    :param sink_snv: Path to write the final SNV parquet. Required when `batch` is set.
    :param sink_insdel: Path to write the final INS/DEL parquet. Required when `batch` is set.
    :param temp_dir_name: Parent directory for temporary files when `batch` is set. Defaults to the system temp dir.
    :param pav_params: PAV parameters.

    :returns: Tuple of (df_snv, df_insdel) DataFrames when no sink paths are provided, else None.
    """
    # Params
    if pav_params is None:
        pav_params = PavParams()

    debug = pav_params.debug

    score_model = get_score_model(pav_params.align_score_model)

    expr_id_snv = expr.id_snv()
    expr_id_insdel = expr.id_nonsnv()

    varscore_snv = score_model.mismatch(1)

    # Alignment dataframe
    if not isinstance(df_align, pl.LazyFrame):
        df_align = df_align.lazy()

    # Batch configuration
    if batch is True:
        batch_size = DEFAULT_BATCH_SIZE
    elif batch is False or int(batch) <= 0:
        batch_size = 0
    else:
        batch_size = int(batch)

    if batch_size > 0 and (sink_snv is None or sink_insdel is None):
        raise RuntimeError('`sink_snv` and `sink_insdel` must be specified when `batch` is set')

    # Row index for alignment access
    df_align = (
        df_align
        .drop('_index', strict=False)
        .with_row_index('_index')
    )

    # Chromosome list in sorted order (determines output order)
    chrom_list = (
        df_align.select('chrom').unique().sort('chrom').collect().to_series().to_list()
    )

    # Sort orders (include _align_score for tie-breaking; it is dropped before output)
    sort_order_snv = {
        'by': ['pos', 'alt', 'var_score', '_align_score', 'qry_id', 'qry_pos'],
        'descending': [False, False, True, True, False, False],
    }
    sort_order_insdel = {
        'by': ['pos', 'end', '_align_score', 'qry_id', 'qry_pos'],
        'descending': [False, False, True, False, False],
    }

    # Helper functions that build the lazy chain for each variant type.
    # Defined here so a zero-row probe can derive the output schema without hardcoding it.
    def _snv_chain(df_lazy: pl.LazyFrame, seq_ref, seq_qry, qry_shift: int, is_rev: bool) -> pl.LazyFrame:
        df_snv = (
            df_lazy
            .filter(pl.col('op_code') == op.X)
            .with_columns(pl.int_ranges(0, pl.col('op_len')).alias('_offset_pos'))
            .explode('_offset_pos')
            .with_columns(
                (pl.col('pos') + pl.col('_offset_pos')).alias('pos'),
                (
                    pl.col('qry_pos') + pl.col('_offset_pos')  # Shift within coordinates
                    + (pl.col('op_len') - 2 * pl.col('_offset_pos') - 1) * qry_shift  # Invert offset on reverse strand
                ).alias('qry_pos'),
            )
            .with_columns(
                (pl.col('pos') + 1).alias('end'),
                (pl.col('qry_pos') + 1).alias('qry_end'),
            )
            .with_columns(
                pl.lit('SNV').alias('vartype'),
                pl.col('pos').map_elements(
                    lambda pos, seq_ref=seq_ref: seq_ref[pos], return_dtype=pl.String
                ).alias('ref'),
                pl.col('qry_pos').map_elements(
                    lambda pos, seq_qry=seq_qry: seq_qry[pos], return_dtype=pl.String
                ).alias('alt'),
            )
        )
        if is_rev:
            df_snv = df_snv.with_columns(
                pl.col('alt').replace(COMPL_TR_FROM, COMPL_TR_TO).alias('alt'),
            )
        return df_snv.select(
            'chrom', 'pos', 'end',
            expr_id_snv,
            'vartype', 'ref', 'alt',
            'filter',
            'qry_id', 'qry_pos', 'qry_end',
            pl.col('is_rev').alias('qry_rev'),
            pl.lit(CALL_SOURCE).alias('call_source'),
            pl.lit(varscore_snv).alias('var_score'),
            'align_source',
            '_align_score',
        )

    def _insdel_chain(df_lazy: pl.LazyFrame, get_ins_seq, seq_ref) -> pl.LazyFrame:
        df_ins = (
            df_lazy
            .filter(pl.col('op_code') == op.I)
            .with_columns(
                (pl.col('pos') + 1).alias('end'),
                pl.lit('INS').alias('vartype'),
                pl.col('op_len').alias('varlen'),
                pl.struct('qry_pos', 'qry_end').map_elements(get_ins_seq, return_dtype=pl.String).alias('seq'),
                (
                    pl.col('op_len')
                    .map_elements(score_model.gap, return_dtype=pl.Float64)
                    .cast(pl.Float32)
                    .alias('var_score')
                ),
            )
        )
        df_del = (
            df_lazy
            .filter(pl.col('op_code') == op.D)
            .with_columns(
                (pl.col('qry_pos') + 1).alias('qry_end'),
                pl.lit('DEL').alias('vartype'),
                pl.col('op_len').alias('varlen'),
                pl.struct('pos', 'end').map_elements(
                    lambda coords, seq_ref=seq_ref: seq_ref[coords['pos']:coords['end']], return_dtype=pl.String
                ).alias('seq'),
                (
                    pl.col('op_len')
                    .map_elements(score_model.gap, return_dtype=pl.Float64)
                    .cast(pl.Float32)
                    .alias('var_score')
                ),
            )
        )
        return pl.concat([df_ins, df_del]).select(
            'chrom', 'pos', 'end',
            expr_id_insdel,
            'vartype', 'varlen',
            'filter',
            'qry_id', 'qry_pos', 'qry_end',
            pl.col('is_rev').alias('qry_rev'),
            pl.lit(CALL_SOURCE).alias('call_source'),
            'var_score',
            'align_source',
            'seq',
            '_align_score',
        )

    # Run the chains on a zero-row probe DataFrame to derive the correct empty-output schemas.
    # This ensures that when df_align is empty (or a chromosome has no variants), the returned
    # DataFrames have exactly the same schema as when variants are present. Any change to the
    # select() calls inside the chain functions automatically propagates here.
    _probe = pl.DataFrame({
        'op_code': pl.Series([], dtype=pl.Int64),
        'op_len': pl.Series([], dtype=pl.Int64),
        'chrom': pl.Series([], dtype=pl.String),
        'pos': pl.Series([], dtype=pl.Int64),
        'end': pl.Series([], dtype=pl.Int64),
        'qry_pos': pl.Series([], dtype=pl.Int64),
        'qry_end': pl.Series([], dtype=pl.Int64),
        'qry_id': pl.Series([], dtype=pl.String),
        'is_rev': pl.Series([], dtype=pl.Boolean),
        'filter': pl.Series([], dtype=pl.List(pl.String)),
        'align_source': pl.Series([], dtype=pl.List(pl.Int32)),
        '_align_score': pl.Series([], dtype=pl.Float32),
    }).lazy()

    _dummy = lambda _: None  # noqa: E731  (placeholder; never called on 0 rows)

    _empty_snv = schema.cast(
        _snv_chain(_probe, _dummy, _dummy, 0, False).collect(),
        schema.VARIANT,
    ).drop('_align_score')

    _empty_insdel = schema.cast(
        _insdel_chain(_probe, _dummy, _dummy).collect(),
        schema.VARIANT,
    ).drop('_align_score')

    # Per-chromosome sorted result frames (LazyFrame, one entry per chromosome)
    final_snv: list[pl.LazyFrame] = []
    final_insdel: list[pl.LazyFrame] = []

    with (
        (TempDirContainer(temp_dir_name, prefix='call_intra_') if batch_size > 0 else NullContext()) as temp_dir,
        LRUSequenceCache(ref_fa_filename, 1) as ref_cache,
        LRUSequenceCache(qry_fa_filename, 10) as qry_cache,
    ):
        for chrom in chrom_list:
            if debug:
                print(f'Intra-alignment discovery: {chrom}')

            seq_ref = ref_cache[chrom]

            # Collect this chromosome's rows once — avoids O(N²) LazyFrame re-execution per alignment
            df_chrom = (
                df_align
                .filter(pl.col('chrom') == chrom)
                .sort('qry_id')
                .collect()
            )

            chrom_snv: list[pl.DataFrame] = []
            chrom_insdel: list[pl.DataFrame] = []

            for row in df_chrom.iter_rows(named=True):
                index = row['_index']
                qry_id = row['qry_id']
                is_rev = row['is_rev']
                pos = row['pos']

                if debug:
                    print(f'* {chrom}: index={index}, qry_id={qry_id}, is_rev={is_rev}, pos={pos}')

                seq_qry = qry_cache[qry_id]
                qry_len = len(seq_qry)
                qry_shift = 1 if is_rev else 0

                if is_rev:
                    def get_ins_seq(coords, seq_qry=seq_qry):
                        return str(
                            Bio.Seq.Seq(seq_qry[coords['qry_pos']:coords['qry_end']]).reverse_complement()
                        )
                else:
                    def get_ins_seq(coords, seq_qry=seq_qry):
                        return seq_qry[coords['qry_pos']:coords['qry_end']]

                # Extract ops to numpy and compute per-op reference/query coordinates
                op_arr = op.row_to_arr(row)
                n_ops = len(op_arr)

                adv_ref = op_arr[:, 1] * np.isin(op_arr[:, 0], op.ADV_REF_ARR)
                adv_qry = op_arr[:, 1] * np.isin(op_arr[:, 0], op.ADV_QRY_ARR)

                ref_pos_arr = np.cumsum(adv_ref) - adv_ref + pos
                qry_pos_arr = np.cumsum(adv_qry) - adv_qry

                if is_rev:
                    qry_pos_col = qry_len - (qry_pos_arr + adv_qry)
                    qry_end_col = qry_len - qry_pos_arr
                else:
                    qry_pos_col = qry_pos_arr
                    qry_end_col = qry_pos_arr + adv_qry

                df = pl.DataFrame({
                    'op_code': op_arr[:, 0],
                    'op_len': op_arr[:, 1],
                    'chrom': pl.Series([chrom] * n_ops, dtype=pl.String),
                    'pos': ref_pos_arr,
                    'end': ref_pos_arr + adv_ref,
                    'qry_pos': qry_pos_col,
                    'qry_end': qry_end_col,
                    'qry_id': pl.Series([qry_id] * n_ops, dtype=pl.String),
                    'is_rev': pl.Series([is_rev] * n_ops, dtype=pl.Boolean),
                    'filter': pl.Series([row['filter']] * n_ops, dtype=pl.List(pl.String)),
                    'align_source': pl.Series([[row['align_index']]] * n_ops, dtype=pl.List(pl.Int32)),
                    '_align_score': pl.Series([row['score']] * n_ops, dtype=pl.Float32),
                }).lazy()

                df_snv, df_insdel = pl.collect_all([
                    _snv_chain(df, seq_ref, seq_qry, qry_shift, is_rev),
                    _insdel_chain(df, get_ins_seq, seq_ref),
                ])

                chrom_snv.append(df_snv)
                chrom_insdel.append(df_insdel)

            # Sort chromosome results and either write to a temp file or keep in memory
            df_chrom_snv = (
                schema.cast(pl.concat(chrom_snv).sort(**sort_order_snv).drop('_align_score'), schema.VARIANT)
                if chrom_snv else _empty_snv
            )
            df_chrom_insdel = (
                schema.cast(pl.concat(chrom_insdel).sort(**sort_order_insdel).drop('_align_score'), schema.VARIANT)
                if chrom_insdel else _empty_insdel
            )

            if batch_size > 0:
                temp_snv_file = temp_dir.next(prefix='snv_', suffix='.parquet')
                temp_insdel_file = temp_dir.next(prefix='insdel_', suffix='.parquet')
                df_chrom_snv.write_parquet(temp_snv_file)
                df_chrom_insdel.write_parquet(temp_insdel_file)
                final_snv.append(pl.scan_parquet(temp_snv_file))
                final_insdel.append(pl.scan_parquet(temp_insdel_file))
            else:
                final_snv.append(df_chrom_snv.lazy())
                final_insdel.append(df_chrom_insdel.lazy())

        # Build final lazy frames (chromosomes already in sorted order from chrom_list)
        lf_snv = pl.concat(final_snv) if final_snv else _empty_snv.lazy()
        lf_insdel = pl.concat(final_insdel) if final_insdel else _empty_insdel.lazy()

        # Sink or return — must happen inside the `with` block so temp files are still accessible
        if batch_size > 0:
            pl.collect_all([
                lf_snv.sink_parquet(Path(sink_snv), lazy=True),
                lf_insdel.sink_parquet(Path(sink_insdel), lazy=True),
            ])

            return None

        result = pl.collect_all([lf_snv, lf_insdel])

        if sink_snv is not None:
            result[0].write_parquet(sink_snv)

        if sink_insdel is not None:
            result[1].write_parquet(sink_insdel)

        return result[0], result[1]


def variant_tables_inv(
        df_align: pl.DataFrame | pl.LazyFrame,
        df_flag: pl.DataFrame | pl.LazyFrame,
        ref_fa_filename: str | Path,
        qry_fa_filename: str | Path,
        df_ref_fai: pl.DataFrame,
        df_qry_fai: pl.DataFrame,
        pav_params: Optional[PavParams] = None,
) -> pl.DataFrame:
    """Call intra-alignment inversions.

    :param df_align: Alignment table.
    :param df_flag: Regions flagged for intra-alignment inversion signatures.
    :param ref_fa_filename: Reference FASTA file name.
    :param qry_fa_filename: Assembly FASTA file name.
    :param df_ref_fai: Reference sequence lengths.
    :param df_qry_fai: Query sequence lengths.
    :param pav_params: PAV parameters.

    :returns: Table of inversion variants.
    """
    # Params
    if pav_params is None:
        pav_params = PavParams()

    # Tables
    if isinstance(df_align, pl.LazyFrame):
        df_align = df_align.collect()

    if isinstance(df_flag, pl.LazyFrame):
        df_flag = df_flag.collect()

    if isinstance(df_ref_fai, pl.LazyFrame):
        df_ref_fai = df_ref_fai.collect()

    if isinstance(df_qry_fai, pl.LazyFrame):
        df_qry_fai = df_qry_fai.collect()

    # Supporting objects
    k_util = agglovar.kmer.util.KmerUtil(pav_params.inv_k_size)

    align_lift = AlignLift(df_align, df_qry_fai)

    kde_model = KdeTruncNorm(
        pav_params.inv_kde_bandwidth, pav_params.inv_kde_trunc_z, pav_params.inv_kde_func
    )

    # Create variant tables
    inv_schema = {col: type_ for col, type_ in schema.VARIANT.items() if col in set(get_inv_row().keys())}

    variant_table_list = []

    log_file = None

    for row in df_flag.iter_rows(named=True):
        region_flag = Region(
            chrom=row['chrom'], pos=row['pos'], end=row['end'],
            pos_align_index=row['align_index'], end_align_index=row['align_index']
        )

        inv_row = try_intra_region(
            region_flag=region_flag,
            ref_fa_filename=ref_fa_filename,
            qry_fa_filename=qry_fa_filename,
            df_ref_fai=df_ref_fai,
            df_qry_fai=df_qry_fai,
            align_lift=align_lift,
            pav_params=pav_params,
            k_util=k_util,
            kde_model=kde_model,
            stop_on_lift_fail=True,
            log_file=log_file,
        )

        if inv_row is not None:
            variant_table_list.append(inv_row)

    df_inv = (
        pl.from_dicts(
            variant_table_list,
            schema=inv_schema,
        )
        .with_columns(
            expr.id_nonsnv().alias('id'),
            pl.lit(CALL_SOURCE).alias('call_source')
        )
        .join(
            df_align.select(['align_index', 'filter']),
            left_on=pl.col('align_source').list.first(),
            right_on='align_index',
            how='left'
        )
    )

    return schema.cast(df_inv, schema.VARIANT)


def variant_flag_inv(
        df_snv: pl.DataFrame | pl.LazyFrame,
        df_insdel: pl.DataFrame | pl.LazyFrame,
        df_ref_fai: pl.DataFrame | pl.LazyFrame,
        df_qry_fai: pl.DataFrame | pl.LazyFrame,
        pav_params: Optional[PavParams] = None,
) -> pl.DataFrame:
    """Flag regions with potential intra-alignment inversions.

    When alignments are pushed through an inversions without splitting into multiple records (i.e. FWD->REV->FWD
    alignment pattern), they leave traces of matching INS & DEL variants and clusters of SNV and indels. This function
    identifies inversion-candidate regions based on these signatures.

    :param df_align: Alignment table.
    :param df_snv: SNV table.
    :param df_insdel: INS/DEL table.
    :param df_ref_fai: Reference sequence lengths.
    :param df_qry_fai: Query sequence lengths.
    :param pav_params: PAV parameters.

    :returns: A table of inversion candidate loci.
    """
    # Params
    if pav_params is None:
        pav_params = PavParams()

    # Tables
    if isinstance(df_snv, pl.DataFrame):
        df_snv = df_snv.lazy()

    if isinstance(df_insdel, pl.DataFrame):
        df_insdel = df_insdel.lazy()

    if isinstance(df_ref_fai, pl.DataFrame):
        df_ref_fai = df_ref_fai.lazy()

    if isinstance(df_qry_fai, pl.DataFrame):
        df_qry_fai = df_qry_fai.lazy()

    df_snv = schema.cast(df_snv, schema.VARIANT)
    df_insdel = schema.cast(df_insdel, schema.VARIANT)

    return (
        cluster_table(
            df_snv=df_snv,
            df_insdel=df_insdel,
            df_ref_fai=df_ref_fai,
            df_qry_fai=df_qry_fai,
            pav_params=pav_params,
        )
        .filter(pl.col('flag') != ['CLUSTER_SNV'])  # Ignore SNV-only clusters
    )
