"""
Call variants by CIGAR string.
"""

import Bio.Seq
import Bio.SeqIO
import numpy as np
import pandas as pd
import pysam

import svpoplib

from . import align
from . import call

#
# Definitions
#

# Tag variants called with this source
CALL_SOURCE = 'CIGAR'


def make_insdel_snv_calls(df_align, ref_fa_name, qry_fa_name, hap, version_id=True, debug=False):
    """
    Parse variants from CIGAR strings.

    :param df_align: Post-cut BED of read alignments.
    :param ref_fa_name: Reference FASTA file name.
    :param qry_fa_name: Assembly FASTA file name.
    :param hap: String identifying the haplotype ("h1", "h2").
    :param version_id: Version duplicate variant IDs if `True`. If `False`, duplicate IDs may be written, and a
        subsequent step should apply versioning.
    :param debug: Extra debugging checks if `True`.

    :return: A tuple of two dataframes, one for insertions and deletions (SV and indel), and one for SNVs.
    """


    df_align = df_align.sort_values(['#CHROM', 'QRY_ID'], ignore_index=True)

    df_insdel_list = list()
    df_snv_list = list()

    seq_ref = None        # Current reference sequence
    seq_ref_upper = None  # Current reference sequence, upper-case
    seq_ref_name = None   # Current reference contig name

    seq_qry = None        # Current query sequence
    seq_qry_upper = None  # Current query sequence, upper-case
    seq_qry_name = None   # Current query name
    seq_qry_len = None    # Current query length
    seq_qry_rev = None    # Aligned query was reverse-complemented

    # Define arrays of operations consuming reference and query bases
    # op_adv_ref_arr = np.array([align.op.EQ, align.op.X, align.op.D])
    # op_adv_qry_arr = np.array([align.op.EQ, align.op.X, align.op.I, align.op.S, align.op.H])
    # op_var_arr = np.array([align.op.X, align.op.I, align.op.D])

    COL_OP_CODE = 0
    COL_OP_LEN = 1
    COL_REF_POS = 2
    COL_QRY_POS = 3
    COL_OP_INDEX = 4

    with (pysam.FastaFile(ref_fa_name) as ref_fa, pysam.FastaFile(qry_fa_name) as qry_fa):

        # Parse alignment records
        for row_index, row in df_align.iterrows():

            # Save alignment fields
            is_rev = row['IS_REV']
            strand = '-' if is_rev else '+'
            align_index = row['INDEX']
            filter = row['FILTER'] if 'filter' in row else 'PASS'

            # Load reference and tig sequences
            if seq_ref_name is None or row['#CHROM'] != seq_ref_name:
                seq_ref_name = str(row['#CHROM'])
                seq_ref = ref_fa.fetch(seq_ref_name)

                seq_ref_upper = seq_ref.upper()

            if seq_qry_name is None or row['QRY_ID'] != seq_qry_name or is_rev != seq_qry_rev:
                seq_qry_name = str(row['QRY_ID'])
                seq_qry = qry_fa.fetch(seq_qry_name)
                seq_qry_len = len(seq_qry)

                if is_rev:
                    seq_qry = str(Bio.Seq.Seq(seq_qry).reverse_complement())

                seq_qry_rev = is_rev

                seq_qry_upper = seq_qry.upper()

            # Get alignment operations and starting positions
            op_arr = align.op.op_arr_add_coords(
                op_arr=align.op.cigar_as_array(row['CIGAR']),
                pos_ref=row['POS'],
                add_index=True
            )

            # adv_ref_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], op_adv_ref_arr)
            # adv_qry_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], op_adv_qry_arr)
            #
            # ref_pos_arr = np.cumsum(adv_ref_arr) - adv_ref_arr + row['POS']
            # qry_pos_arr = np.cumsum(adv_qry_arr) - adv_qry_arr
            #
            # # Check for zero-length operations (no operation or bad CIGAR length)
            # if np.any((adv_ref_arr + adv_qry_arr) == 0):
            #     no_op_arr = op_arr[(adv_ref_arr + adv_qry_arr == 0) & (op_arr[:, 1] > 0)]
            #     no_len_arr = op_arr[op_arr[:, 1] == 0]
            #
            #     op_set = ', '.join(sorted(set(no_op_arr[:, 0].astype(str))))
            #     len_set = ', '.join(sorted(set(no_len_arr[:, 0].astype(str))))
            #
            #     if op_set:
            #         raise RuntimeError(f'Unexpected operations in CIGAR string at {align_index}: operation code(s) "{op_set}"')
            #
            #     if len_set:
            #         raise RuntimeError(f'Zero-length operations CIGAR string at {align_index}: operation code(s) "{len_set}"')
            #
            # op_arr = np.concatenate([
            #     op_arr,
            #     np.expand_dims(ref_pos_arr, axis=1),
            #     np.expand_dims(qry_pos_arr, axis=1),
            #     np.expand_dims(np.arange(op_arr.shape[0]), axis=1)
            # ], axis=1)

            op_arr = op_arr[np.isin(op_arr[:, 0], align.op.VAR_ARR)]

            for index in range(len(op_arr)):

                op_code = op_arr[index, COL_OP_CODE]
                op_len = op_arr[index, COL_OP_LEN]
                op_index = op_arr[index, COL_OP_INDEX]

                if op_code == align.op.X:
                    # Call SNV(s)

                    for i in range(op_len):

                        # Get position and bases
                        pos_ref = op_arr[index, COL_REF_POS] + i
                        pos_qry = op_arr[index, COL_QRY_POS] + i

                        base_ref = seq_ref[pos_ref]
                        base_qry = seq_qry[pos_qry]

                        if debug:
                            if base_ref.upper() == base_qry.upper():
                                raise RuntimeError(f'Expected base mismatch at {align_index} (op_code={align.op.OP_CHAR_FUNC(op_code)}, op_len={op_arr[index, COL_OP_LEN]}, op_index={op_index}): ref={base_ref}, qry={base_qry}')

                        # Query coordinates
                        if is_rev:
                            pos_qry = seq_qry_len - pos_qry - 1

                        if debug:
                            base_qry_exp = qry_fa.fetch(seq_qry_name, pos_qry, pos_qry + 1)

                            if is_rev:
                                base_qry_exp = str(Bio.Seq.Seq(base_qry_exp).reverse_complement())

                            if base_qry != base_qry_exp:
                                raise RuntimeError(f'Coordinate mismatch at {align_index} (op_code={align.op.OP_CHAR_FUNC(op_code)}, op_len={op_arr[index, COL_OP_LEN]}, op_index={op_index}): base_qry={base_qry}, expected={base_qry_exp}')

                        # Add variant
                        df_snv_list.append(
                            pd.Series(
                                [
                                    seq_ref_name, pos_ref, pos_ref + 1,
                                    np.nan, 'SNV', 1,
                                    base_ref, base_qry,
                                    filter,
                                    hap,
                                    seq_qry_name, pos_qry, pos_qry + 1, strand,
                                    0,
                                    align_index,
                                    CALL_SOURCE
                                ],
                                index=[
                                    '#CHROM', 'POS', 'END',
                                    'ID', 'SVTYPE', 'SVLEN',
                                    'REF', 'ALT',
                                    'FILTER',
                                    'HAP',
                                    'QRY_ID', 'QRY_POS', 'QRY_END', 'QRY_STRAND',
                                    'CI',
                                    'ALIGN_INDEX',
                                    'CALL_SOURCE'
                                ]
                            )
                        )

                elif op_code == align.op.I:
                    # Call INS

                    pos_qry = op_arr[index, COL_QRY_POS]
                    pos_ref = op_arr[index, COL_REF_POS]

                    seq = seq_qry[pos_qry:(pos_qry + op_len)]
                    seq_upper = seq.upper()

                    # Left shift
                    left_shift = (
                        np.min([
                            op_arr[index - 1, COL_OP_LEN],
                            call.left_homology(pos_ref - 1, seq_ref_upper, seq_upper)  # SV/breakpoint upstream homology
                        ])
                    ) if index > 0 and op_arr[index - 1, COL_OP_CODE] == align.op.EQ else 0

                    if debug and left_shift < 0:
                        raise RuntimeError(f'Negative left shift at {align_index} (op_code={op_code}({align.op.OP_CHAR_FUNC(op_code)}), op_len={op_arr[index, COL_OP_LEN]}, op_index={op_index}): left_shift={left_shift}')

                    pos_ref -= left_shift
                    pos_qry -= left_shift

                    end_qry = pos_qry + op_len

                    if left_shift != 0:
                        seq = seq_qry[pos_qry:(pos_qry + op_len)]
                        seq_upper = seq.upper()

                    # Get positions in the original coordinates
                    if is_rev:
                        end_qry = seq_qry_len - pos_qry
                        pos_qry = end_qry - op_len

                    if debug:
                        seq_exp = qry_fa.fetch(seq_qry_name, pos_qry, end_qry)

                        if is_rev:
                            seq_exp = str(Bio.Seq.Seq(seq_exp).reverse_complement())

                        if seq != seq_exp:
                            raise RuntimeError(f'Sequence mismatch at {align_index} (op_code={align.op.OP_CHAR_FUNC(op_code)}, op_len={op_arr[index, COL_OP_LEN]}, op_index={op_index}): seq="{seq}", expected="{seq_exp}"')

                    # Find breakpoint homology
                    hom_ref_l = call.left_homology(pos_ref - 1, seq_ref_upper, seq_upper)
                    hom_ref_r = call.right_homology(pos_ref, seq_ref_upper, seq_upper)

                    hom_qry_l = call.left_homology(pos_qry - 1, seq_qry_upper, seq_upper)
                    hom_qry_r = call.right_homology(end_qry, seq_qry_upper, seq_upper)

                    # Add variant
                    df_insdel_list.append(
                        pd.Series(
                            [
                                seq_ref_name, pos_ref, pos_ref + 1,
                                np.nan, 'INS', op_len,
                                filter,
                                hap,
                                seq_qry_name, pos_qry, end_qry, strand,
                                0,
                                align_index,
                                left_shift, f'{hom_ref_l},{hom_ref_r}', f'{hom_qry_l},{hom_qry_r}',
                                CALL_SOURCE,
                                seq
                            ],
                            index=[
                                '#CHROM', 'POS', 'END',
                                'ID', 'SVTYPE', 'SVLEN',
                                'FILTER',
                                'HAP',
                                'QRY_ID', 'QRY_POS', 'QRY_END', 'QRY_STRAND',
                                'CI',
                                'ALIGN_INDEX',
                                'LEFT_SHIFT', 'HOM_REF', 'HOM_TIG',
                                'CALL_SOURCE',
                                'SEQ'
                            ]
                        )
                    )

                elif op_code == align.op.D:
                    # Call DEL

                    pos_qry = op_arr[index, COL_QRY_POS]
                    pos_ref = op_arr[index, COL_REF_POS]

                    # Get sequence
                    seq = seq_ref[pos_ref:(pos_ref + op_len)]
                    seq_upper = seq.upper()

                    # Left shift
                    left_shift = (
                        np.min([
                            op_arr[index - 1, COL_OP_LEN],
                            call.left_homology(pos_ref - 1, seq_ref_upper, seq_upper)  # SV/breakpoint upstream homology
                        ])
                    ) if index > 0 and op_arr[index - 1, COL_OP_CODE] == align.op.EQ else 0

                    if debug and left_shift < 0:
                        raise RuntimeError(f'Negative left shift at {align_index} (op_code={op_code}({align.op.OP_CHAR_FUNC(op_code)}), op_len={op_arr[index, COL_OP_LEN]}, op_index={op_index}): left_shift={left_shift}')

                    pos_ref -= left_shift
                    end_ref = pos_ref + op_len

                    pos_qry -= left_shift

                    # Get positions in the original coordinates
                    if is_rev:
                        pos_qry = seq_qry_len - pos_qry

                    if debug:
                        seq_exp = ref_fa.fetch(seq_ref_name, pos_ref, end_ref)

                        if seq != seq_exp:
                            raise RuntimeError(f'Sequence mismatch at {align_index} (op_code={align.op.OP_CHAR_FUNC(op_code)}, op_len={op_arr[index, COL_OP_LEN]}, op_index={op_index}): seq="{seq}", expected="{seq_exp}"')

                    # Find breakpoint homology
                    hom_ref_l = call.left_homology(pos_ref - 1, seq_ref_upper, seq_upper)
                    hom_ref_r = call.right_homology(pos_ref + 1, seq_ref_upper, seq_upper)

                    hom_qry_l = call.left_homology(pos_qry - 1, seq_qry_upper, seq_upper)
                    hom_qry_r = call.right_homology(pos_qry, seq_qry_upper, seq_upper)

                    # Add variant
                    df_insdel_list.append(pd.Series(
                        [
                            seq_ref_name, pos_ref, end_ref,
                            np.nan, 'DEL', op_len,
                            filter,
                            hap,
                            seq_qry_name, pos_qry, pos_qry + 1, strand,
                            0,
                            align_index,
                            left_shift, f'{hom_ref_l},{hom_ref_r}', f'{hom_qry_l},{hom_qry_r}',
                            CALL_SOURCE,
                            seq
                        ],
                        index=[
                            '#CHROM', 'POS', 'END',
                            'ID', 'SVTYPE', 'SVLEN',
                            'FILTER',
                            'HAP',
                            'QRY_ID', 'QRY_POS', 'QRY_END', 'QRY_STRAND',
                            'CI',
                            'ALIGN_INDEX',
                            'LEFT_SHIFT', 'HOM_REF', 'HOM_TIG',
                            'CALL_SOURCE',
                            'SEQ'
                        ]
                    ))

    # Merge tables
    if len(df_snv_list) > 0:
        df_snv = pd.concat(df_snv_list, axis=1).T

        df_snv.sort_values(['#CHROM', 'POS', 'END', 'ID'], inplace=True)

        df_snv['ID'] = svpoplib.variant.get_variant_id(df_snv, apply_version=version_id)

    else:
        df_snv = pd.DataFrame(
            [],
            columns=[
                '#CHROM', 'POS', 'END',
                'ID', 'SVTYPE', 'SVLEN',
                'REF', 'ALT',
                'FILTER',
                'HAP',
                'QRY_ID', 'QRY_POS', 'QRY_END', 'QRY_STRAND',
                'CI',
                'ALIGN_INDEX',
                'CALL_SOURCE'
            ]
        )


    if len(df_insdel_list) > 0:
        df_insdel = pd.concat(df_insdel_list, axis=1).T

        df_insdel.sort_values(['#CHROM', 'POS', 'END', 'ID'], inplace=True)

        df_insdel['ID'] = svpoplib.variant.get_variant_id(df_insdel, apply_version=version_id)

    else:
        df_insdel = pd.DataFrame(
            [],
            columns=[
                '#CHROM', 'POS', 'END',
                'ID', 'SVTYPE', 'SVLEN',
                'FILTER',
                'HAP',
                'QRY_ID', 'QRY_POS', 'QRY_END', 'QRY_STRAND',
                'CI',
                'ALIGN_INDEX',
                'LEFT_SHIFT', 'HOM_REF', 'HOM_TIG',
                'CALL_SOURCE',
                'SEQ'
            ]
        )

    # Return tables
    return df_snv, df_insdel
