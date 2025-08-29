"""
Intra-alignment variant calling from alignment operations (derived from CIGAR strings).
"""

import agglovar
import os
from typing import Optional

import Bio.Seq
import Bio.SeqIO
import numpy as np
import polars as pl

from .. import align as pavcall_align
from .. import kde as pavcall_kde
from .. import params as pavcall_params
from .. import schema as pavcall_schema
from .. import seq as pavcall_seq

from . import expr as pavcall_call_expr
from . import inv as pavcall_call_inv

# Tag variants called with this source
CALL_SOURCE = 'INTRA'
"""str: Variant call source column value."""


def variant_tables(
        df_align: pl.DataFrame | pl.LazyFrame,
        ref_fa_filename: str,
        qry_fa_filename: str,
        df_ref_fai: pl.DataFrame,
        df_qry_fai: pl.DataFrame,
        temp_dir_name: Optional[str] = None,
        pav_params: Optional[pavcall_params.PavParams] = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Call variants from alignment operations.

    Calls SNVs, INS/DEL (indel and SV), and inversions (INVs) as three separate tables.

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

    Args:
        df_align: Assembly alignments before alignment trimming.
        ref_fa_filename: Reference FASTA file name.
        qry_fa_filename: Assembly FASTA file name.
        df_ref_fai: Reference sequence lengths.
        df_qry_fai: Query sequence lengths.
        temp_dir_name: Temporary directory name for variant tables (one parquet file per chromosome) or None to retain
            all variants in memory.
        pav_params: PAV parameters.

    Returns:
        Tuple of three LazyFrames: SNV variants, INS/DEL variants, and INV variants.
    """

    # Params
    if pav_params is None:
        pav_params = pavcall_params.PavParams()

    debug = pav_params.debug

    # Alignment dataframe
    if not isinstance(df_align, pl.LazyFrame):
        df_align = df_align.lazy()

    chrom_list = df_align.select('chrom').unique().sort('chrom').collect().to_series().to_list()

    # Structures for intra-alignment INVs
    k_util = agglovar.kmer.util.KmerUtil(pav_params.inv_k_size)

    align_lift = pavcall_align.lift.AlignLift(df_align.collect(), df_qry_fai)

    kde_model = pavcall_kde.KdeTruncNorm(
        pav_params.inv_kde_bandwidth, pav_params.inv_kde_trunc_z, pav_params.inv_kde_func
    )

    # Temp directory
    if temp_dir_name is not None and not os.path.isdir(temp_dir_name):
        raise ValueError(f'Temporary directory does not exist or is not a directory: {temp_dir_name}')

    # Create variant tables
    with (
        pavcall_seq.LRUSequenceCache(ref_fa_filename, 1) as ref_cache,
        pavcall_seq.LRUSequenceCache(qry_fa_filename, 10) as qry_cache,
    ):

        chrom_table_list = {'snv': [], 'insdel': [], 'inv': []}

        for chrom in chrom_list:
            if debug:
                print(f'Intra-alignment discovery: {chrom}')

            temp_file_name = {
                'snv': os.path.join(temp_dir_name, f'snv_{chrom}.parquet'),
                'insdel': os.path.join(temp_dir_name, f'insdel_{chrom}.parquet'),
                'inv': os.path.join(temp_dir_name, f'inv_{chrom}.parquet'),
            } if temp_dir_name is not None else None

            seq_ref = ref_cache[chrom]

            df_chrom_list = {'snv': [], 'insdel': [], 'inv': []}

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
                row_list = {'snv': [], 'insdel': [], 'inv': []}

                # Augment operation array
                op_arr = pavcall_align.op.row_to_arr(row)

                adv_ref_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], pavcall_align.op.ADV_REF_ARR)
                adv_qry_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], pavcall_align.op.ADV_QRY_ARR)

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
                for index in np.where(op_arr[:, 0] == pavcall_align.op.X)[0]:
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
                                str(pavcall_align.op.OP_CHAR_FUNC(op_code)), op_len, op_index, is_rev, base_ref, base_qry
                            )
                        )

                        # Query coordinates
                        if debug:
                            base_qry_exp = seq_qry_org[pos_qry].upper()

                            if is_rev:
                                base_qry_exp = str(Bio.Seq.Seq(base_qry_exp).reverse_complement())

                            assert base_qry == base_qry_exp, (
                                'Expected base does not mach (reverse-complement logic error?): '
                                'Operation (op_code="%s", op_len=%d, op_index=%d, is_rev=%s): '
                                'base=%s, expected=%s' % (
                                    str(pavcall_align.op.OP_CHAR_FUNC(op_code)),
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
                for index in np.where((op_arr[:, 0] == pavcall_align.op.I) | (op_arr[:, 0] == pavcall_align.op.D))[0]:
                    op_code, op_len, op_pos_ref, op_pos_qry, op_index = op_arr[index]

                    assert op_code in {pavcall_align.op.I, pavcall_align.op.D}, (
                        'Unexpected alignment operation at alignment index %d: %s' % (align_index, str(pavcall_align.op.OP_CHAR_FUNC(op_code)))
                    )

                    pos_qry = (len(seq_qry) - op_pos_qry - op_len) if is_rev else op_pos_qry

                    if op_code == pavcall_align.op.I:
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

                    elif op_code == pavcall_align.op.D:
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
                    if debug and op_code == pavcall_align.op.I:
                        seq_exp = seq_qry_org[pos_qry:pos_qry + op_len].upper()

                        if is_rev:
                            seq_exp = str(Bio.Seq.Seq(seq_exp).reverse_complement())

                        assert seq.upper() == seq_exp, (
                            'Expected sequence does not mach (reverse-complement logic error?): '
                            'Operation (op_code="%s", op_len=%d, op_index=%d, is_rev=%s): ' % (
                                str(pavcall_align.op.OP_CHAR_FUNC(op_code)),
                                op_len, op_index, is_rev
                            )
                        )

                # Collect SNV and INS/DEL tables for this alignment record (row)
                df_snv = (
                    pl.DataFrame(
                        row_list['snv'],
                        orient='row',
                        schema={
                            key: pavcall_schema.VARIANT[key]
                                for key in ('pos', 'align_index', 'qry_pos', 'ref', 'alt')
                        }
                    )
                    .lazy()
                    .with_columns(
                        pl.lit(chrom).cast(pavcall_schema.VARIANT['chrom']).alias('chrom'),
                        (pl.col('pos') + 1).cast(pavcall_schema.VARIANT['end']).alias('end'),
                        pl.lit(row['qry_id']).cast(pavcall_schema.VARIANT['qry_id']).alias('qry_id'),
                        (pl.col('qry_pos') + 1).cast(pavcall_schema.VARIANT['qry_end']).alias('qry_end'),
                    )
                    .with_columns(
                        pavcall_call_expr.id_snv().alias('id')
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
                            key: pavcall_schema.VARIANT[key] for key in (
                                'pos', 'end', 'vartype', 'align_index', 'qry_pos', 'qry_end', 'varlen', 'seq'
                            )
                        }
                    )
                    .lazy()
                    .with_columns(
                        pl.lit(chrom).cast(pavcall_schema.VARIANT['chrom']).alias('chrom'),
                        pl.lit(row['qry_id']).cast(pavcall_schema.VARIANT['qry_id']).alias('qry_id'),
                    )
                    .with_columns(
                        pavcall_call_expr.id_nonsnv().alias('id')
                    )
                    .collect()
                    .lazy()
                )

                df_chrom_list['insdel'].append(df_insdel)

                # Identify clusters of variants for INS/DEL
                df_flag = (
                    pavcall_call_inv.cluster_merge(
                        df_snv=df_snv,
                        df_insdel=df_insdel,
                        df_ref_fai=df_ref_fai,
                        df_qry_fai=df_qry_fai,
                        pav_params=pav_params,
                    )
                )











            # Save chromosome
            df_snv = (
                pl.concat(df_chrom_list['snv'])
                .with_columns(
                    chrom=pl.lit(chrom).cast(pavcall_schema.VARIANT['chrom']),
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
                .sort(['pos', 'alt', '_align_score', 'qry_id', 'qry_pos'], descending=[False, False, True, False, False])
                .drop('_align_score')
            )

            df_insdel = (
                pl.concat(df_chrom_list['insdel'])
                .with_columns(
                    chrom=pl.lit(chrom).cast(pavcall_schema.VARIANT['chrom']),
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
                .sort(['pos', 'end', '_align_score', 'qry_id', 'qry_pos'], descending=[False, False, True, False, False])
                .drop('_align_score')
            )

            # Save chromosome-level tables
            if temp_file_name is not None:
                # If using a temporary file, write file and scan it (add to list of LazyFrames to concat)
                df_snv.sink_parquet(temp_file_name['snv'])
                chrom_table_list['snv'].append(pl.scan_parquet(temp_file_name['snv']))

                df_insdel.sink_parquet(temp_file_name['insdel'])
                chrom_table_list['insdel'].append(pl.scan_parquet(temp_file_name['insdel']))

                df_inv.sink_parquet(temp_file_name['inv'])
                chrom_table_list['inv'].append(pl.scan_parquet(temp_file_name['inv']))

            else:
                # if not using a temporary file, save in-memory tables to be concatenated.
                chrom_table_list['snv'].append(df_snv.lazy())
                chrom_table_list['insdel'].append(df_insdel.lazy())
                chrom_table_list['inv'].append(df_inv.lazy())

        # Concat tables
        return (
            pl.concat(chrom_table_list['snv']),
            pl.concat(chrom_table_list['insdel']),
            pl.concat(chrom_table_list['inv'])
        )


# def snv_table(
#         df_align: pl.DataFrame | pl.LazyFrame,
#         ref_fa_filename: str,
#         qry_fa_filename: str,
#         temp_dir_name: Optional[str] = None,
#         debug: bool = False,
# ) -> pl.LazyFrame:
#     """
#     Call variants from alignment operations (derived from CIGAR strings).
#
#     The temporary directory is used to avoid keeping all variant calls in memory and is useful for large SNV callsets,
#     such as species significantly diverged from the reference. Variants are called by chromosome, sorted, and saved to a
#     parquet file in this directory. At the end, they are concatenated together and returned as a sorted LazyFrame. Files
#     in this directory must be retained until the LazyFrame is collected or sunk.
#
#     Variant sort order is chromosome (chrom), position (pos), alternate base (alt), alignment score (highest first,
#     column not retained in variant table), query ID (qry_id), and query position (qry_pos). This ensures variants are
#     sorted in a deterministic way across PAV runs.
#
#     Args:
#         df_align: Assembly alignments before alignment trimming.
#         ref_fa_filename: Reference FASTA file name.
#         qry_fa_filename: Assembly FASTA file name.
#         temp_dir_name: Temporary directory name for variant tables (one parquet file per chromosome) or None to retain
#             all variants in memory.
#         debug: Extra debugging checks if `True`.
#
#     Returns:
#         A sorted LazyFrame of variant calls.
#     """
#
#     if not isinstance(df_align, pl.LazyFrame):
#         df_align = df_align.lazy()
#
#     chrom_list = df_align.select('chrom').unique().sort('chrom').collect().to_series().to_list()
#
#     if temp_dir_name is not None and not os.path.isdir(temp_dir_name):
#         raise ValueError(f'Temporary directory does not exist or is not a directory: {temp_dir_name}')
#
#     with (
#         pavcall_seq.LRUSequenceCache(ref_fa_filename, 1) as ref_cache,
#         pavcall_seq.LRUSequenceCache(qry_fa_filename, 10) as qry_cache,
#     ):
#
#         temp_file_list = list()
#
#         for chrom in chrom_list:
#             if debug:
#                 print(f'Calling SNVs: {chrom}')
#
#             temp_file_name = os.path.join(temp_dir_name, f'snv_{chrom}.parquet') if temp_dir_name is not None else None
#
#             seq_ref = ref_cache[chrom]
#
#             df_chrom_list = []
#
#             for row in (
#                     df_align
#                     .filter(pl.col('chrom') == chrom)
#                     .sort('qry_id')
#                     .collect()
#                     .iter_rows(named=True)
#             ):
#
#                 # Query sequence
#                 seq_qry = qry_cache[(row['qry_id'], row['is_rev'])]
#                 seq_qry_org = qry_cache[row['qry_id']] if debug else None
#
#                 # Augment operation array
#                 op_arr = pavcall_align.op.row_to_arr(row)
#
#                 adv_ref_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], pavcall_align.op.ADV_REF_ARR)
#                 adv_qry_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], pavcall_align.op.ADV_QRY_ARR)
#
#                 ref_pos_arr = np.cumsum(adv_ref_arr) - adv_ref_arr + row['pos']
#                 qry_pos_arr = np.cumsum(adv_qry_arr) - adv_qry_arr
#
#                 op_arr = np.concatenate([
#                     op_arr,
#                     np.expand_dims(ref_pos_arr, axis=1),
#                     np.expand_dims(qry_pos_arr, axis=1),
#                     np.expand_dims(np.arange(op_arr.shape[0]), axis=1)
#                 ], axis=1)
#
#                 op_arr = op_arr[op_arr[:, 0] == pavcall_align.op.X]
#
#                 # Save frequently-used fields
#                 align_index = row['align_index']
#                 is_rev = row['is_rev']
#
#                 # Call SNVs
#                 row_list = []
#
#                 for index in range(op_arr.shape[0]):
#                     op_code, op_len, op_pos_ref, op_pos_qry, op_index = op_arr[index]
#
#                     for i in range(op_len):
#
#                         # Get position and bases
#                         pos_ref = op_pos_ref + i
#                         pos_qry = (len(seq_qry) - (op_pos_qry + i) - 1) if is_rev else (op_pos_qry + i)
#
#                         base_ref = seq_ref[pos_ref]
#                         base_qry = seq_qry[op_pos_qry + i]
#
#                         assert base_ref.upper() != base_qry.upper(), (
#                             'Bases match at alignment mismatch site: '
#                             'Operation (op_code="%s", op_len=%d, op_index=%d, is_rev=%s): '
#                             'ref=%s, qry=%s' % (
#                                 str(pavcall_align.op.OP_CHAR_FUNC(op_code)), op_len, op_index, is_rev, base_ref, base_qry
#                             )
#                         )
#
#                         # Query coordinates
#                         if debug:
#                             base_qry_exp = seq_qry_org[pos_qry].upper()
#
#                             if is_rev:
#                                 base_qry_exp = str(Bio.Seq.Seq(base_qry_exp).reverse_complement())
#
#                             assert base_qry == base_qry_exp, (
#                                 'Expected base does not mach (reverse-complement logic error?): '
#                                 'Operation (op_code="%s", op_len=%d, op_index=%d, is_rev=%s): '
#                                 'base=%s, expected=%s' % (
#                                     str(pavcall_align.op.OP_CHAR_FUNC(op_code)),
#                                     op_len, op_index, is_rev, base_qry, base_qry_exp
#                                 )
#                             )
#
#                         # Add variant
#                         row_list.append((
#                             pos_ref,
#                             align_index,
#                             pos_qry,
#                             base_ref,
#                             base_qry
#                         ))
#
#                 df_chrom_list.append(
#                     pl.DataFrame(
#                         row_list,
#                         orient='row',
#                         schema={
#                             key: pavcall_schema.VARIANT[key]
#                                 for key in ('pos', 'align_index', 'qry_pos', 'ref', 'alt')
#                         }
#                     )
#                 )
#
#             # Save chromosome
#             df_snv = (
#                 pl.concat(df_chrom_list)
#                 .lazy()
#                 .with_columns(
#                     chrom=pl.lit(chrom).cast(pavcall_schema.VARIANT['chrom']),
#                 )
#                 .join(
#                     (
#                         df_align
#                         .select(
#                             pl.col(['align_index', 'qry_id', 'filter']),
#                             pl.col('score').alias('_align_score')
#                         )
#                     ),
#                     on='align_index',
#                     how='left'
#                 )
#                 .sort(['pos', 'alt', '_align_score', 'qry_id', 'qry_pos'], descending=[False, False, True, False, False])
#                 .drop('_align_score')
#             )
#
#             if temp_file_name is not None:
#                 df_snv.sink_parquet(temp_file_name)
#                 temp_file_list.append(temp_file_name)
#             else:
#                 temp_file_list.append(df_snv)
#
#     # Concat and add deferred columns
#     df_snv = (
#         pl.concat(
#             [pl.scan_parquet(temp_file_name) for temp_file_name in temp_file_list]
#                 if temp_dir_name is not None
#                 else temp_file_list  # LazyFrames in list
#         )
#         .with_columns(
#             (pl.col('pos') + 1).cast(pavcall_schema.VARIANT['end']).alias('end'),
#             (pl.col('qry_pos') + 1).cast(pavcall_schema.VARIANT['qry_end']).alias('qry_end'),
#             pavcall_call_expr.id_snv().alias('id'),
#             (pl.lit(CALL_SOURCE)).cast(pavcall_schema.VARIANT['call_source']).alias('call_source'),
#         )
#     )
#
#     return df_snv.select(
#         [key for key in pavcall_schema.VARIANT.keys() if key in set(df_snv.collect_schema().names())]
#     )
#
#
# def insdel_table(
#         df_align: pl.DataFrame | pl.LazyFrame,
#         ref_fa_filename: str,
#         qry_fa_filename: str,
#         temp_dir_name: Optional[str] = None,
#         debug: bool = False,
# ) -> pl.LazyFrame:
#     """
#     Call variants from alignment operations (derived from CIGAR strings).
#
#     The temporary directory is used to avoid keeping all variant calls in memory and is useful for large SNV callsets,
#     such as species significantly diverged from the reference. Variants are called by chromosome, sorted, and saved to a
#     parquet file in this directory. At the end, they are concatenated together and returned as a sorted LazyFrame. Files
#     in this directory must be retained until the LazyFrame is collected or sunk.
#
#     Variant sort order is chromosome (chrom), position (pos), end position (end), alignment score (highest first,
#     column not retained in variant table), query ID (qry_id), and query position (qry_pos). This ensures variants are
#     sorted in a deterministic way across PAV runs.
#
#     Args:
#         df_align: Assembly alignments before alignment trimming.
#         ref_fa_filename: Reference FASTA file name.
#         qry_fa_filename: Assembly FASTA file name.
#         temp_dir_name: Temporary directory name for variant tables (one parquet file per chromosome) or None to retain
#             all variants in memory.
#         debug: Extra debugging checks if `True`.
#
#     Returns:
#         A sorted LazyFrame of variant calls.
#     """
#
#     if not isinstance(df_align, pl.LazyFrame):
#         df_align = df_align.lazy()
#
#     chrom_list = df_align.select('chrom').unique().sort('chrom').collect().to_series().to_list()
#
#     if temp_dir_name is not None and not os.path.isdir(temp_dir_name):
#         raise ValueError(f'Temporary directory does not exist or is not a directory: {temp_dir_name}')
#
#     with (
#         pavcall_seq.LRUSequenceCache(ref_fa_filename, 1) as ref_cache,
#         pavcall_seq.LRUSequenceCache(qry_fa_filename, 10) as qry_cache,
#     ):
#
#         temp_file_list = list()
#
#         for chrom in chrom_list:
#             if debug:
#                 print(f'Calling indels: {chrom}')
#
#             temp_file_name = os.path.join(temp_dir_name, f'insdel_{chrom}.parquet') if temp_dir_name is not None else None
#
#             seq_ref = ref_cache[chrom]
#
#             df_chrom_list = []
#
#             for row in (
#                     df_align
#                     .filter(pl.col('chrom') == chrom)
#                     .sort('qry_id')
#                     .collect()
#                     .iter_rows(named=True)
#             ):
#
#                 # Query sequence
#                 seq_qry = qry_cache[(row['qry_id'], row['is_rev'])]
#                 seq_qry_org = qry_cache[row['qry_id']] if debug else None
#
#                 # Augment operation array
#                 op_arr = pavcall_align.op.row_to_arr(row)
#
#                 adv_ref_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], pavcall_align.op.ADV_REF_ARR)
#                 adv_qry_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], pavcall_align.op.ADV_QRY_ARR)
#
#                 ref_pos_arr = np.cumsum(adv_ref_arr) - adv_ref_arr + row['pos']
#                 qry_pos_arr = np.cumsum(adv_qry_arr) - adv_qry_arr
#
#                 op_arr = np.concatenate([
#                     op_arr,
#                     np.expand_dims(ref_pos_arr, axis=1),
#                     np.expand_dims(qry_pos_arr, axis=1),
#                     np.expand_dims(np.arange(op_arr.shape[0]), axis=1)
#                 ], axis=1)
#
#                 op_arr = op_arr[
#                     (op_arr[:, 0] == pavcall_align.op.I) | (op_arr[:, 0] == pavcall_align.op.D)
#                 ]
#
#                 # Save frequently-used fields
#                 align_index = row['align_index']
#                 is_rev = row['is_rev']
#
#                 # Call insertions and deletions
#                 row_list = []
#
#                 for index in range(op_arr.shape[0]):
#                     op_code, op_len, op_pos_ref, op_pos_qry, op_index = op_arr[index]
#
#                     assert op_code in {pavcall_align.op.I, pavcall_align.op.D}, (
#                         'Unexpected alignment operation at alignment index %d: %s' % (align_index, str(pavcall_align.op.OP_CHAR_FUNC(op_code)))
#                     )
#
#                     pos_qry = (len(seq_qry) - op_pos_qry - op_len) if is_rev else op_pos_qry
#
#                     if op_code == pavcall_align.op.I:
#                         seq = seq_qry[op_pos_qry:op_pos_qry + op_len]
#
#                         row_list.append((
#                             op_pos_ref,
#                             op_pos_ref + 1,
#                             'INS',
#                             align_index,
#                             pos_qry,
#                             pos_qry + op_len,
#                             op_len,
#                             seq
#                         ))
#
#                     elif op_code == pavcall_align.op.D:
#                         seq = seq_ref[op_pos_ref:op_pos_ref + op_len]
#
#                         row_list.append((
#                             op_pos_ref,
#                             op_pos_ref + op_len,
#                             'DEL',
#                             align_index,
#                             pos_qry,
#                             pos_qry + 1,
#                             op_len,
#                             seq
#                         ))
#
#                     # Query coordinates
#                     if debug and op_code == pavcall_align.op.I:
#                         seq_exp = seq_qry_org[pos_qry:pos_qry + op_len].upper()
#
#                         if is_rev:
#                             seq_exp = str(Bio.Seq.Seq(seq_exp).reverse_complement())
#
#                         assert seq.upper() == seq_exp, (
#                             'Expected sequence does not mach (reverse-complement logic error?): '
#                             'Operation (op_code="%s", op_len=%d, op_index=%d, is_rev=%s): ' % (
#                                 str(pavcall_align.op.OP_CHAR_FUNC(op_code)),
#                                 op_len, op_index, is_rev
#                             )
#                         )
#
#                 df_chrom_list.append(
#                     pl.DataFrame(
#                         row_list,
#                         orient='row',
#                         schema={
#                             key: pavcall_schema.VARIANT[key] for key in (
#                                 'pos', 'end', 'vartype', 'align_index', 'qry_pos', 'qry_end', 'varlen', 'seq'
#                             )
#                         }
#                     )
#                 )
#
#             # Save chromosome
#             df_insdel = (
#                 pl.concat(df_chrom_list)
#                 .lazy()
#                 .with_columns(
#                     chrom=pl.lit(chrom).cast(pavcall_schema.VARIANT['chrom']),
#                 )
#                 .join(
#                     (
#                         df_align
#                         .select(
#                             pl.col(['align_index', 'qry_id', 'filter']),
#                             pl.col('score').alias('_align_score')
#                         )
#                     ),
#                     on='align_index',
#                     how='left'
#                 )
#                 .sort(['pos', 'end', '_align_score', 'qry_id', 'qry_pos'], descending=[False, False, True, False, False])
#                 .drop('_align_score')
#             )
#
#             if temp_file_name is not None:
#                 df_insdel.sink_parquet(temp_file_name)
#                 temp_file_list.append(temp_file_name)
#             else:
#                 temp_file_list.append(df_insdel)
#
#     # Concat and add deferred columns
#     df_insdel = (
#         pl.concat(
#             [pl.scan_parquet(temp_file_name) for temp_file_name in temp_file_list]
#                 if temp_dir_name is not None
#                 else temp_file_list  # LazyFrames in list
#         )
#         .with_columns(
#             (pl.col('pos') + 1).cast(pavcall_schema.VARIANT['end']).alias('end'),
#             (pl.col('qry_pos') + 1).cast(pavcall_schema.VARIANT['qry_end']).alias('qry_end'),
#             pavcall_call_expr.id_nonsnv().alias('id'),
#             (pl.lit(CALL_SOURCE)).cast(pavcall_schema.VARIANT['call_source']).alias('call_source'),
#         )
#     )
#
#     return df_insdel.select(
#         [key for key in pavcall_schema.VARIANT.keys() if key in set(df_insdel.collect_schema().names())]
#     )
