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
import numpy as np
import polars as pl

from .. import schema

from ..align import op

from ..align.lift import AlignLift
from ..inv import cluster_table, get_inv_row, try_intra_region
from ..kde import KdeTruncNorm
from ..region import Region
from ..params import PavParams
from ..seq import LRUSequenceCache

from . import expr


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

    # Alignment dataframe
    if not isinstance(df_align, pl.LazyFrame):
        df_align = df_align.lazy()

    chrom_list = df_align.select('chrom').unique().sort('chrom').collect().to_series().to_list()

    # Temp directory
    if temp_dir_name is not None and not os.path.isdir(temp_dir_name):
        raise ValueError(f'Temporary directory does not exist or is not a directory: {temp_dir_name}')

    # Temporary schema - Leave align_index as an integer until the end (after align table join)
    build_schema = schema.VARIANT | {'align_index': pl.Int32}

    # Create variant tables
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

            for row in (
                    df_align
                    .filter(pl.col('chrom') == chrom)
                    .sort('qry_id')
                    .collect()
                    .iter_rows(named=True)
            ):

                # Query sequence
                seq_qry = qry_cache[(row['qry_id'], row['is_rev'])]
                seq_qry_org = qry_cache[row['qry_id']] if debug else None

                # List for collecting variants for this row
                row_list = {'snv': [], 'insdel': []}

                # Augment operation array
                op_arr = op.row_to_arr(row)

                adv_ref_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], op.ADV_REF_ARR)
                adv_qry_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], op.ADV_QRY_ARR)

                ref_pos_arr = np.cumsum(adv_ref_arr) - adv_ref_arr + row['pos']
                qry_pos_arr = np.cumsum(adv_qry_arr) - adv_qry_arr

                op_arr = np.concatenate([
                    op_arr,
                    np.expand_dims(ref_pos_arr, axis=1),
                    np.expand_dims(qry_pos_arr, axis=1),
                    np.expand_dims(np.arange(op_arr.shape[0]), axis=1)
                ], axis=1)

                # Save frequently-used fields
                align_index = row['align_index']
                is_rev = row['is_rev']

                # Call SNV
                for index in np.where(op_arr[:, 0] == op.X)[0]:
                    op_code, op_len, op_pos_ref, op_pos_qry, op_index = op_arr[index]

                    for i in range(op_len):

                        # Get position and bases
                        pos_ref = op_pos_ref + i
                        pos_qry = (len(seq_qry) - (op_pos_qry + i) - 1) if is_rev else (op_pos_qry + i)

                        base_ref = seq_ref[pos_ref]
                        base_qry = seq_qry[op_pos_qry + i]

                        assert base_ref.upper() != base_qry.upper(), (
                            'Bases match at alignment mismatch site: '
                            'Operation (op_code="%s", op_len=%d, op_index=%d, is_rev=%s): '
                            'ref=%s, qry=%s' % (
                                str(op.OP_CHAR_FUNC(op_code)), op_len, op_index, is_rev, base_ref, base_qry
                            )
                        )

                        # Query coordinates
                        if debug:
                            base_qry_exp = seq_qry_org[pos_qry].upper()

                            if is_rev:
                                base_qry_exp = str(Bio.Seq.Seq(base_qry_exp).reverse_complement())

                            assert base_qry == base_qry_exp, (
                                'Expected base does not match (reverse-complement logic error?): '
                                'Operation (op_code="%s", op_len=%d, op_index=%d, is_rev=%s): '
                                'base=%s, expected=%s' % (
                                    str(op.OP_CHAR_FUNC(op_code)),
                                    op_len, op_index, is_rev, base_qry, base_qry_exp
                                )
                            )

                        # Add variant
                        row_list['snv'].append((
                            pos_ref,
                            align_index,
                            pos_qry,
                            base_ref,
                            base_qry
                        ))

                # Call INS/DEL
                for index in np.where((op_arr[:, 0] == op.I) | (op_arr[:, 0] == op.D))[0]:
                    op_code, op_len, op_pos_ref, op_pos_qry, op_index = op_arr[index]

                    assert op_code in {op.I, op.D}, (
                        'Unexpected alignment operation at alignment index %d: %s' % (
                            align_index, str(op.OP_CHAR_FUNC(op_code))
                        )
                    )

                    pos_qry = (len(seq_qry) - op_pos_qry - op_len) if is_rev else op_pos_qry

                    if op_code == op.I:
                        seq = seq_qry[op_pos_qry:op_pos_qry + op_len]

                        row_list['insdel'].append((
                            op_pos_ref,
                            op_pos_ref + 1,
                            'INS',
                            align_index,
                            pos_qry,
                            pos_qry + op_len,
                            op_len,
                            seq
                        ))

                    elif op_code == op.D:
                        seq = seq_ref[op_pos_ref:op_pos_ref + op_len]

                        row_list['insdel'].append((
                            op_pos_ref,
                            op_pos_ref + op_len,
                            'DEL',
                            align_index,
                            pos_qry,
                            pos_qry + 1,
                            op_len,
                            seq
                        ))

                    # Query coordinates
                    if debug and op_code == op.I:
                        seq_exp = seq_qry_org[pos_qry:pos_qry + op_len].upper()

                        if is_rev:
                            seq_exp = str(Bio.Seq.Seq(seq_exp).reverse_complement())

                        assert seq.upper() == seq_exp, (
                            'Expected sequence does not match (reverse-complement logic error?): '
                            'Operation (op_code="%s", op_len=%d, op_index=%d, is_rev=%s): ' % (
                                str(op.OP_CHAR_FUNC(op_code)),
                                op_len, op_index, is_rev
                            )
                        )

                # Collect SNV and INS/DEL tables for this alignment record (row)
                # TODO: Defer chrom and id to chromosome-level variants
                # TODO: Add qry_rev
                df_snv = (
                    pl.DataFrame(
                        row_list['snv'],
                        orient='row',
                        schema={
                            key: build_schema[key]
                            for key in ('pos', 'align_index', 'qry_pos', 'ref', 'alt')
                        }
                    )
                    .lazy()
                    .with_columns(
                        pl.lit(chrom).cast(build_schema['chrom']).alias('chrom'),
                        (pl.col('pos') + 1).cast(build_schema['end']).alias('end'),
                        pl.lit(row['qry_id']).cast(build_schema['qry_id']).alias('qry_id'),
                        (pl.col('qry_pos') + 1).cast(build_schema['qry_end']).alias('qry_end'),
                    )
                    .with_columns(
                        expr.id_snv().alias('id')
                    )
                    .collect()
                    .lazy()
                )

                df_chrom_list['snv'].append(df_snv)

                df_insdel = (
                    pl.DataFrame(
                        row_list['insdel'],
                        orient='row',
                        schema={
                            key: build_schema[key] for key in (
                                'pos', 'end', 'vartype', 'align_index', 'qry_pos', 'qry_end', 'varlen', 'seq'
                            )
                        }
                    )
                    .lazy()
                    .with_columns(
                        pl.lit(chrom).cast(build_schema['chrom']).alias('chrom'),
                        pl.lit(row['qry_id']).cast(build_schema['qry_id']).alias('qry_id'),
                    )
                    .with_columns(
                        expr.id_nonsnv().alias('id')
                    )
                    .collect()
                    .lazy()
                )

                df_chrom_list['insdel'].append(df_insdel)

            # Save chromosome
            df_snv = (
                pl.concat(df_chrom_list['snv'])
                .with_columns(
                    pl.lit(chrom).cast(build_schema['chrom']).alias('chrom'),
                )
                .join(
                    (
                        df_align
                        .select(
                            pl.col(['align_index', 'qry_id', 'filter']),
                            pl.col('score').alias('_align_score')
                        )
                    ),
                    on='align_index',
                    how='left'
                )
                .with_columns(  # align_index back to a list
                    pl.concat_list(['align_index']).alias('align_index')
                )
                .sort(
                    ['pos', 'alt', '_align_score', 'qry_id', 'qry_pos'],
                    descending=[False, False, True, False, False]
                )
                .drop('_align_score')
            )

            df_insdel = (
                pl.concat(df_chrom_list['insdel'])
                .with_columns(
                    pl.lit(chrom).cast(build_schema['chrom']).alias('chrom'),
                )
                .join(
                    (
                        df_align
                        .select(
                            pl.col(['align_index', 'qry_id', 'filter']),
                            pl.col('score').alias('_align_score')
                        )
                    ),
                    on='align_index',
                    how='left'
                )
                .with_columns(  # align_index back to a list
                    pl.concat_list(['align_index']).alias('align_index')
                )
                .sort(
                    ['pos', 'end', '_align_score', 'qry_id', 'qry_pos'],
                    descending=[False, False, True, False, False]
                )
                .drop('_align_score')
            )

            # Save chromosome-level tables
            if temp_file_name is not None:
                # If using a temporary file, write file and scan it (add to list of LazyFrames to concat)
                df_snv.sink_parquet(temp_file_name['snv'])
                chrom_table_list['snv'].append(pl.scan_parquet(temp_file_name['snv']))

                df_insdel.sink_parquet(temp_file_name['insdel'])
                chrom_table_list['insdel'].append(pl.scan_parquet(temp_file_name['insdel']))

            else:
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
    col_set.add('align_index')

    table_schema = {col: type_ for col, type_ in schema.VARIANT.items() if col in col_set}

    return (
        pl.from_dicts(variant_table_list, schema=table_schema)
        .with_columns(
            expr.id_nonsnv().alias('id'),
            pl.lit(CALL_SOURCE).alias('call_source')
        )
        .join(
            df_align.select(['align_index', 'filter']),
            left_on=pl.col('align_index').list.first(),
            right_on='align_index',
            how='left'
        )
        .select(list(table_schema.keys()))
    )


def variant_flag_inv(
        df_align: pl.DataFrame | pl.LazyFrame,
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
    if isinstance(df_align, pl.DataFrame):
        df_align = df_align.lazy()

    if isinstance(df_snv, pl.DataFrame):
        df_snv = df_snv.lazy()

    if isinstance(df_insdel, pl.DataFrame):
        df_insdel = df_insdel.lazy()

    if isinstance(df_ref_fai, pl.DataFrame):
        df_ref_fai = df_ref_fai.lazy()

    if isinstance(df_qry_fai, pl.DataFrame):
        df_qry_fai = df_qry_fai.lazy()

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
