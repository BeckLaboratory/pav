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
    'variant_tables_snv_insdel',
    'variant_tables_inv',
    'variant_flag_inv'
]

import agglovar
import os
from typing import Optional

import Bio.Seq
import Bio.SeqIO
import polars as pl

from .. import schema

from ..align import op

from ..align.lift import AlignLift
from ..align.score import ScoreModel, get_score_model
from ..inv import cluster_table, get_inv_row, try_intra_region
from ..kde import KdeTruncNorm
from ..region import Region
from ..params import PavParams
from ..seq import LRUSequenceCache

from . import expr
from .util import COMPL_TR_FROM, COMPL_TR_TO


# Tag variants called with this source
CALL_SOURCE: str = 'INTRA'
"""Variant call source column value."""


def variant_tables_snv_insdel(
        df_align: pl.DataFrame | pl.LazyFrame,
        ref_fa_filename: str,
        qry_fa_filename: str,
        temp_dir_name: Optional[str] = None,
        pav_params: Optional[PavParams] = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Call variants from alignment operations.

    Calls variants in two separate tables, SNVs in the first, and INS/DEL (including indel and SV) in the second.

    Each chromosome is processed separately. If a temporary directory is defined, then the three variant call tables
    for each chromosome is written to the temporary directory location (N files = 3 * M chromosomes). For divergent
    species (e.g. diverse mouse species or nonhuman primates vs a human reference), this can reduce memory usage. If
    a temporary directory is not defined, then the tables are held in memory.

    The temporary tables (in memory or on disk) are sorted by all fields except "chrom" (see below), so a sorted
    table is achieved by concatenating the temporary tables in chromosomal order. Temporary tables on disk are
    parquet files so they can be concatenated without excessive memory demands.

    The LazyFrames returned by this function are constructed by concatenating the temporary tables in chromosomal
    order. To write directly to disk, sink these LazyFrames to a final table. To create an in-memory table, collect
    them.

    Variant sort order is chromosome (chrom), position (pos), alternate base (alt, SNVs) or end position (end, non-SNV),
    alignment score (highest first, column not retained in variant table), query ID (qry_id), and query position
    (qry_pos). This ensures variants are sorted in a deterministic way across PAV runs.

    :param df_align: Assembly alignments before alignment trimming.
    :param ref_fa_filename: Reference FASTA file name.
    :param qry_fa_filename: Assembly FASTA file name.
    :param temp_dir_name: Temporary directory name for variant tables (one parquet file per chromosome) or None to
        retain oall variants in memory.
    :param pav_params: PAV parameters.

    :returns: Tuple of three LazyFrames: SNV variants, INS/DEL variants, and INV variants.
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

    chrom_list = df_align.select('chrom').unique().sort('chrom').collect().to_series().to_list()

    # Temp directory
    if temp_dir_name is not None and not os.path.isdir(temp_dir_name):
        raise ValueError(f'Temporary directory does not exist or is not a directory: {temp_dir_name}')

    # Create variant tables
    df_align = (
        df_align
        .drop('_index', strict=False)
        .with_row_index('_index')
    )

    with (
        LRUSequenceCache(ref_fa_filename, 1) as ref_cache,
        LRUSequenceCache(qry_fa_filename, 10) as qry_cache,
    ):
        chrom_table_list = {'snv': [], 'insdel': []}

        for chrom in chrom_list:
            if debug:
                print(f'Intra-alignment discovery: {chrom}')

            temp_file_name = {
                'snv': os.path.join(temp_dir_name, f'snv_{chrom}.parquet'),
                'insdel': os.path.join(temp_dir_name, f'insdel_{chrom}.parquet'),
            } if temp_dir_name is not None else None

            seq_ref = ref_cache[chrom]

            df_chrom_list = {'snv': [], 'insdel': []}

            for index, qry_id, is_rev, pos in (
                    df_align
                    .filter(pl.col('chrom') == chrom)
                    .sort('qry_id')
                    .select('_index', 'qry_id', 'is_rev', 'pos')
                    .collect()
                    .iter_rows()
            ):
                if debug:
                    print(f'* {chrom}: index={index}, qry_id={qry_id}, is_rev={is_rev}, pos={pos}')

                # Query sequence
                seq_qry = qry_cache[qry_id]

                qry_len = len(seq_qry)

                qry_shift = 1 if is_rev else 0

                if is_rev:
                    get_ins_seq = lambda coords: str(Bio.Seq.Seq(seq_qry[coords['qry_pos']:coords['qry_end']]).reverse_complement())
                else:
                    get_ins_seq = lambda coords: seq_qry[coords['qry_pos']:coords['qry_end']]

                # Get alignment indices
                df = (
                    df_align
                    .filter(pl.col('_index') == index)
                    .with_columns(
                        pl.col('align_ops').struct.field('op_code'),
                        pl.col('align_ops').struct.field('op_len'),
                    )
                    .drop('align_ops')
                    .explode(['op_code', 'op_len'])
                    .with_columns(
                        pl.when(pl.col('op_code').is_in(op.ADV_QRY_ARR))
                        .then(pl.col('op_len'))
                        .otherwise(0)
                        .alias('qry_len'),
                        pl.when(pl.col('op_code').is_in(op.ADV_REF_ARR))
                        .then(pl.col('op_len'))
                        .otherwise(0)
                        .alias('ref_len'),
                    )
                    .with_columns(
                        (pl.col('ref_len').cum_sum() + pos).alias('end'),
                        (pl.col('qry_len').cum_sum()).alias('qry_end'),
                    )
                    .with_columns(
                        pl.col('end').shift(1, fill_value=pos).alias('pos'),
                        pl.col('qry_end').shift(1, fill_value=0).alias('qry_pos'),
                        pl.concat_list('align_index').alias('align_source'),
                        pl.col('score').alias('_align_score')
                    )
                )

                if is_rev:
                    df = (
                        df
                        .with_columns(
                            (qry_len - (pl.col('qry_end'))).alias('qry_pos'),
                            (qry_len - pl.col('qry_pos')).alias('qry_end'),
                        )
                    )

                # Call SNV
                df_snv = (
                    df
                    .filter(pl.col('op_code') == op.X)

                    # Expand multi-base SNVs (MNVs) to individual SNV calls
                    .with_columns(
                        pl.int_ranges(0, pl.col('op_len')).alias('_offset_pos')
                    )
                    .explode('_offset_pos')
                    .with_columns(
                        (pl.col('pos') + pl.col('_offset_pos')).alias('pos'),
                        (
                            pl.col('qry_pos') + pl.col('_offset_pos')  # Shift within coordinates
                            + (pl.col('op_len') - 2 * pl.col('_offset_pos') - 1) * qry_shift  # If on reverse coordinates, invert coordinates within multi-base mismatch alignments
                        ).alias('qry_pos'),
                    )
                    .with_columns(
                        (pl.col('pos') + 1).alias('end'),
                        (pl.col('qry_pos') + 1).alias('qry_end'),
                    )

                    # Complete SNV variant table
                    .with_columns(
                        pl.lit('SNV').alias('vartype'),
                        pl.col('pos').map_elements(lambda pos: seq_ref[pos]).alias('ref'),
                        pl.col('qry_pos').map_elements(lambda pos: seq_qry[pos]).alias('alt'),
                    )
                )

                if is_rev:
                    df_snv = df_snv.with_columns(
                        pl.col('alt').replace(COMPL_TR_FROM, COMPL_TR_TO).alias('alt'),
                    )

                df_snv = (
                    df_snv
                    .select(
                        'chrom', 'pos', 'end',
                        expr_id_snv,
                        'vartype',
                        'ref', 'alt',
                        'filter',
                        'qry_id', 'qry_pos', 'qry_end',
                        pl.col('is_rev').alias('qry_rev'),
                        pl.lit(CALL_SOURCE).alias('call_source'),
                        pl.lit(varscore_snv).alias('var_score'),
                        'align_source',
                        '_align_score',
                    )
                    .with_row_index('_index')
                )

                # Call INS/DEL
                df_ins = (
                    df
                    .filter(pl.col('op_code') == op.I)
                    .with_columns(
                        (pl.col('pos') + 1).alias('end'),
                        pl.lit('INS').alias('vartype'),
                        pl.col('op_len').alias('varlen'),
                        (
                            pl.struct('qry_pos', 'qry_end')
                            .map_elements(get_ins_seq, return_dtype=pl.String)
                        ).alias('seq'),
                        pl.col('op_len').map_elements(score_model.gap).alias('var_score'),
                    )
                )

                df_del = (
                    df
                    .filter(pl.col('op_code') == op.D)
                    .with_columns(
                        (pl.col('qry_pos') + 1).alias('qry_end'),
                        pl.lit('DEL').alias('vartype'),
                        pl.col('op_len').alias('varlen'),
                        (
                            pl.struct('pos', 'end')
                            .map_elements(lambda coords: seq_ref[coords['pos']:coords['end']], return_dtype=pl.String)
                        ).alias('seq'),
                        pl.col('op_len').map_elements(score_model.gap).alias('var_score'),
                    )
                )

                df_insdel = (
                    pl.concat([df_ins, df_del])
                    .sort(
                        ['pos', 'end', '_align_score', 'qry_id', 'qry_pos'],
                        descending=[False, False, True, False, False]
                    )
                    .select(
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
                )

                df_snv = df_snv.collect().lazy()
                df_insdel = df_insdel.collect().lazy()

                # Append to chromosome list
                df_chrom_list['snv'].append(df_snv)
                df_chrom_list['insdel'].append(df_insdel)

            df_snv = schema.cast(
                (
                    pl.concat(df_chrom_list['snv'])
                    .sort(
                        ['pos', 'alt', 'var_score', '_align_score', 'qry_id', 'qry_pos'],
                        descending=[False, False, True, True, False, False]
                    )
                    .drop('_align_score')
                ),
                schema.VARIANT
            )

            df_insdel = schema.cast(
                (
                    pl.concat(df_chrom_list['insdel'])
                    .sort(
                        ['pos', 'var_score', '_align_score', 'qry_id', 'qry_pos'],
                        descending=[False, True, True, False, False]
                    )
                    .drop('_align_score')
                ),
                schema.VARIANT
            )

            # Save chromosome-level tables
            if temp_file_name is not None:
                # If using a temporary file, write file and scan it (add to list of LazyFrames to concat)

                write_snv = df_snv.sink_parquet(temp_file_name['snv'], lazy=True)
                write_insdel = df_insdel.sink_parquet(temp_file_name['insdel'], lazy=True)

                pl.collect_all([write_snv, write_insdel])

                chrom_table_list['snv'].append(pl.scan_parquet(temp_file_name['snv']))
                chrom_table_list['insdel'].append(pl.scan_parquet(temp_file_name['insdel']))

            else:
                df_snv, df_insdel = pl.collect_all([df_snv, df_insdel,])

                # if not using a temporary file, save in-memory tables to be concatenated.
                chrom_table_list['snv'].append(df_snv.lazy())
                chrom_table_list['insdel'].append(df_insdel.lazy())

        # Concat tables
        return (
            pl.concat(chrom_table_list['snv']),
            pl.concat(chrom_table_list['insdel'])
        )


def variant_tables_inv(
        df_align: pl.DataFrame | pl.LazyFrame,
        df_flag: pl.DataFrame | pl.LazyFrame,
        ref_fa_filename: str,
        qry_fa_filename: str,
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
    variant_table_list = []

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
            log_file=None,
        )

        if inv_row is not None:
            inv_row['align_index'] = [row['align_index']]

            variant_table_list.append(pl.DataFrame(inv_row))

    col_set = set(get_inv_row().keys())

    col_set.add('filter')
    col_set.add('align_source')

    table_schema = {col: type_ for col, type_ in schema.VARIANT.items() if col in col_set}

    df_inv = (
        pl.from_dicts(variant_table_list, schema=table_schema)
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
        .select(list(table_schema.keys()))
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
