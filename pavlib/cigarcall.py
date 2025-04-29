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
from . import call as pav_call

#
# Definitions
#

# Tag variants called with this source
CALL_SOURCE = 'CIGAR'


def make_insdel_snv_calls(df_align, ref_fa_name, qry_fa_name, hap, version_id=True):
    """
    Parse variants from CIGAR strings.

    :param df_align: Post-cut BED of read alignments.
    :param ref_fa_name: Reference FASTA file name.
    :param qry_fa_name: Assembly FASTA file name.
    :param hap: String identifying the haplotype ("h1", "h2").
    :param version_id: Version duplicate variant IDs if `True`. If `False`, duplicate IDs may be written, and a
        subsequent step should apply versioning.

    :return: A tuple of two dataframes, one for insertions and deletions (SV and indel), and one for SNVs.
    """

    df_insdel_list = list()
    df_snv_list = list()

    seq_ref = None       # Current reference sequence
    seq_ref_name = None  # Current reference contig name

    seq_qry = None       # Current query sequence
    seq_qry_name = None  # Current query name
    seq_qry_len = None   # Current query length
    seq_qry_rev = None   # Aligned query was reverse-complemented

    # Define arrays of operations consuming reference and query bases
    op_adv_ref_arr = np.array([align.op.EQ, align.op.X, align.op.D])
    op_adv_qry_arr = np.array([align.op.EQ, align.op.X, align.op.I, align.op.S, align.op.H])

    COL_OP_CODE = 0
    COL_OP_LEN = 1
    COL_REF_POS = 2
    COL_QRY_POS = 3

    # Parse alignment records
    for index, row in df_align.iterrows():

        # Get strand
        is_rev = row['IS_REV']
        strand = '-' if is_rev else '+'
        align_index = row['INDEX']

        # Load reference and tig sequences
        if seq_ref_name is None or row['#CHROM'] != seq_ref_name:
            with pysam.FastaFile(ref_fa_name) as ref_fa:
                seq_ref_name = row['#CHROM']
                seq_ref = ref_fa.fetch(str(seq_ref_name))

        if seq_qry_name is None or row['QRY_ID'] != seq_qry_name or is_rev != seq_qry_rev:
            with pysam.FastaFile(qry_fa_name) as tig_fa:
                seq_qry_name = row['QRY_ID']
                seq_qry = tig_fa.fetch(str(seq_qry_name))
                seq_qry_len = len(seq_qry)

                if is_rev:
                    seq_qry = str(Bio.Seq.Seq(seq_qry).reverse_complement())

                seq_qry_rev = is_rev

        seq_ref_upper = seq_ref.upper()
        seq_qry_upper = seq_qry.upper()

        # # Process CIGAR
        # pos_ref = row['POS']
        # pos_qry = 0
        #
        # cigar_index = 0

        # last_op = None
        # last_oplen = 0

        # Get alignment operations and starting positions
        op_arr = align.op.cigar_as_array(row['CIGAR'])

        adv_ref_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], op_adv_ref_arr)
        adv_qry_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], op_adv_qry_arr)

        ref_pos_arr = np.cumsum(adv_ref_arr) - adv_ref_arr + row['POS']
        qry_pos_arr = np.cumsum(adv_qry_arr) - adv_qry_arr

        if np.any((adv_ref_arr + adv_qry_arr) == 0):
            no_op_arr = op_arr[(adv_ref_arr + adv_qry_arr == 0) & (op_arr[:, 1] > 0)]
            no_len_arr = op_arr[op_arr[:, 1] == 0]

            op_set = ', '.join(sorted(set(no_op_arr[:, 0].astype(str))))
            len_set = ', '.join(sorted(set(no_len_arr[:, 0].astype(str))))

            if op_set:
                raise RuntimeError(f'Unexpected operations in CIGAR string at {row["INDEX"]}: operation code(s) "{op_set}"')

            if len_set:
                raise RuntimeError(f'Zero-length operations CIGAR string at {row["INDEX"]}: operation code(s) "{len_set}"')

        op_arr = np.concatenate([op_arr, np.expand_dims(ref_pos_arr, axis=1), np.expand_dims(qry_pos_arr, axis=1)], axis=1)
    #
    #     for oplen, op in align.op.as_tuples(row):
    #         # NOTE: break/continue in this loop will not advance last_op and last_oplen (end of loop)
    #
    #         cigar_index += 1
    #
    #         if op == '=':
    #             pos_ref += oplen
    #             pos_qry += oplen
    #
    #         elif op == 'X':
    #             # Call SNV(s)
    #
    #             for i in range(oplen):
    #
    #                 # Get position and bases
    #                 pos_ref_snv = pos_ref + i
    #                 pos_tig_snv = pos_qry + i
    #
    #                 base_ref = seq_ref[pos_ref_snv]
    #                 base_tig = seq_qry[pos_tig_snv]
    #
    #                 # pos_tig_snv to fwd contig if alignment is reversed
    #                 if is_rev:
    #                     pos_tig_snv = seq_qry_len - pos_tig_snv - 1
    #
    #                 # Add variant
    #                 var_id = f'{seq_ref_name}-{pos_ref_snv + 1}-SNV-{base_ref.upper()}{base_tig.upper()}'
    #
    #                 df_snv_list.append(pd.Series(
    #                     [
    #                         seq_ref_name, pos_ref_snv, pos_ref_snv + 1,
    #                         var_id, 'SNV', 1,
    #                         base_ref, base_tig,
    #                         hap,
    #                         f'{seq_qry_name}:{pos_tig_snv + 1}-{pos_tig_snv + 1}', strand,
    #                         0,
    #                         align_index,
    #                         CALL_SOURCE
    #                     ],
    #                     index=[
    #                         '#CHROM', 'POS', 'END',
    #                         'ID', 'SVTYPE', 'SVLEN',
    #                         'REF', 'ALT',
    #                         'HAP',
    #                         'QRY_REGION', 'QRY_STRAND',
    #                         'CI',
    #                         'ALIGN_INDEX',
    #                         'CALL_SOURCE'
    #                     ]
    #                 ))
    #
    #             # Advance
    #             pos_ref += oplen
    #             pos_qry += oplen
    #
    #         elif op == 'I':
    #             # Call INS
    #
    #             # Get sequence
    #             seq = seq_qry[pos_qry:(pos_qry + oplen)]
    #             seq_upper = seq.upper()
    #
    #             # Left shift
    #             if last_op == '=':
    #                 left_shift = np.min([
    #                     last_oplen,
    #                     pav_call.left_homology(pos_ref - 1, seq_ref_upper, seq_upper)  # SV/breakpoint upstream homology
    #                 ])
    #             else:
    #                 left_shift = 0
    #
    #             sv_pos_ref = pos_ref - left_shift
    #             sv_end_ref = sv_pos_ref + 1
    #             sv_pos_tig = pos_qry - left_shift
    #             sv_end_tig = sv_pos_tig + oplen
    #
    #             if left_shift != 0:
    #                 seq = seq_qry[sv_pos_tig:(sv_pos_tig + oplen)]
    #
    #             # Get positions in the original SV space
    #             # pos_tig_insdel to fwd contig if alignment is reversed
    #             if is_rev:
    #                 end_tig_insdel = seq_qry_len - sv_pos_tig
    #                 pos_tig_insdel = end_tig_insdel - oplen
    #
    #             else:
    #                 pos_tig_insdel = sv_pos_tig
    #                 end_tig_insdel = pos_tig_insdel + oplen
    #
    #             # Find breakpoint homology
    #             seq_upper = seq.upper()
    #
    #             hom_ref_l = pav_call.left_homology(sv_pos_ref - 1, seq_ref_upper, seq_upper)
    #             hom_ref_r = pav_call.right_homology(sv_pos_ref, seq_ref_upper, seq_upper)
    #
    #             hom_tig_l = pav_call.left_homology(sv_pos_tig - 1, seq_tig_upper, seq_upper)
    #             hom_tig_r = pav_call.right_homology(sv_end_tig, seq_tig_upper, seq_upper)
    #
    #             # Add variant
    #             var_id = f'{seq_ref_name}-{sv_pos_ref + 1}-INS-{oplen}'
    #
    #             df_insdel_list.append(pd.Series(
    #                 [
    #                     seq_ref_name, sv_pos_ref, sv_end_ref,
    #                     var_id, 'INS', oplen,
    #                     hap,
    #                     f'{seq_qry_name}:{pos_tig_insdel + 1}-{end_tig_insdel}', strand,
    #                     0,
    #                     align_index,
    #                     left_shift, f'{hom_ref_l},{hom_ref_r}', f'{hom_tig_l},{hom_tig_r}',
    #                     CALL_SOURCE,
    #                     seq
    #                 ],
    #                 index=[
    #                     '#CHROM', 'POS', 'END',
    #                     'ID', 'SVTYPE', 'SVLEN',
    #                     'HAP',
    #                     'QRY_REGION', 'QRY_STRAND',
    #                     'CI',
    #                     'ALIGN_INDEX',
    #                     'LEFT_SHIFT', 'HOM_REF', 'HOM_TIG',
    #                     'CALL_SOURCE',
    #                     'SEQ'
    #                 ]
    #             ))
    #
    #             # Advance
    #             pos_qry += oplen
    #
    #             pass
    #
    #         elif op == 'D':
    #             # Call DEL
    #
    #             # Get sequence
    #             seq = seq_ref[pos_ref:(pos_ref + oplen)]
    #             seq_upper = seq.upper()
    #
    #             # Left shift
    #             if last_op == '=':
    #                 left_shift = np.min([
    #                     last_oplen,
    #                     pav_call.left_homology(pos_ref - 1, seq_ref_upper, seq_upper)  # SV/breakpoint upstream homology
    #                 ])
    #             else:
    #                 left_shift = 0
    #
    #             sv_pos_ref = pos_ref - left_shift
    #             sv_end_ref = sv_pos_ref + oplen
    #             sv_pos_tig = pos_qry - left_shift
    #             sv_end_tig = sv_pos_tig + 1
    #
    #             # Contig position in original coordinates (translate if - strand)
    #             pos_tig_insdel = sv_pos_tig
    #
    #             if is_rev:
    #                 pos_tig_insdel = seq_qry_len - sv_pos_tig
    #
    #             # Find breakpoint homology
    #             seq_upper = seq.upper()
    #
    #             hom_ref_l = pav_call.left_homology(sv_pos_ref - 1, seq_ref_upper, seq_upper)
    #             hom_ref_r = pav_call.right_homology(sv_end_ref, seq_ref_upper, seq_upper)
    #
    #             hom_tig_l = pav_call.left_homology(sv_pos_tig - 1, seq_tig_upper, seq_upper)
    #             hom_tig_r = pav_call.right_homology(sv_pos_tig, seq_tig_upper, seq_upper)
    #
    #             # Add variant
    #             var_id = f'{seq_ref_name}-{pos_ref + 1}-DEL-{oplen}'
    #
    #             df_insdel_list.append(pd.Series(
    #                 [
    #                     seq_ref_name, pos_ref, pos_ref + oplen,
    #                     var_id, 'DEL', oplen,
    #                     hap,
    #                     f'{seq_qry_name}:{pos_tig_insdel + 1}-{pos_tig_insdel + 1}', strand,
    #                     0,
    #                     align_index,
    #                     left_shift, f'{hom_ref_l},{hom_ref_r}', f'{hom_tig_l},{hom_tig_r}',
    #                     CALL_SOURCE,
    #                     seq
    #                 ],
    #                 index=[
    #                     '#CHROM', 'POS', 'END',
    #                     'ID', 'SVTYPE', 'SVLEN',
    #                     'HAP',
    #                     'QRY_REGION', 'QRY_STRAND',
    #                     'CI',
    #                     'ALIGN_INDEX',
    #                     'LEFT_SHIFT', 'HOM_REF', 'HOM_TIG',
    #                     'CALL_SOURCE',
    #                     'SEQ'
    #                 ]
    #             ))
    #
    #             # Advance
    #             pos_ref += oplen
    #
    #             pass
    #
    #         elif op in {'S', 'H'}:
    #             pos_qry += oplen
    #
    #         else:
    #             # Cannot handle CIGAR operation
    #
    #             if op == 'M':
    #                 raise RuntimeError((
    #                     'Illegal operation code in CIGAR string at operation {}: '
    #                     'Alignments must be generated with =/X (not M): '
    #                     'opcode={}, subject={}:{}, query={}:{}, align-index={}'
    #                 ).format(
    #                     cigar_index, op, seq_ref_name, pos_ref, seq_qry_name, pos_qry, row['INDEX']
    #                 ))
    #
    #             else:
    #                 raise RuntimeError((
    #                     'Illegal operation code in CIGAR string at operation {}: '
    #                     'opcode={}, subject={}:{} , query={}:{}, align-index={}'
    #                 ).format(
    #                     cigar_index, op, seq_ref_name, pos_ref, seq_qry_name, pos_qry, row['INDEX']
    #                 ))
    #
    #         # Save last op
    #         last_op = op
    #         last_oplen = oplen
    #
    # # Merge tables
    # if len(df_snv_list) > 0:
    #     df_snv = pd.concat(df_snv_list, axis=1).T
    #
    #     if version_id:
    #         df_snv['ID'] = svpoplib.variant.version_id(df_snv['ID'])
    #
    #     df_snv.sort_values(['#CHROM', 'POS', 'END', 'ID'], inplace=True)
    #
    # else:
    #     df_snv = pd.DataFrame(
    #         [],
    #         columns=[
    #             '#CHROM', 'POS', 'END',
    #             'ID', 'SVTYPE', 'SVLEN',
    #             'REF', 'ALT',
    #             'HAP',
    #             'QRY_REGION', 'QRY_STRAND',
    #             'CI',
    #             'ALIGN_INDEX',
    #             'CALL_SOURCE'
    #         ]
    #     )
    #
    # if len(df_insdel_list) > 0:
    #     df_insdel = pd.concat(df_insdel_list, axis=1).T
    #
    #     if version_id:
    #         df_insdel['ID'] = svpoplib.variant.version_id(df_insdel['ID'])
    #
    #     df_insdel.sort_values(['#CHROM', 'POS', 'END', 'ID'], inplace=True)
    #
    # else:
    #     df_insdel = pd.DataFrame(
    #         [],
    #         columns=[
    #             '#CHROM', 'POS', 'END',
    #             'ID', 'SVTYPE', 'SVLEN',
    #             'HAP',
    #             'QRY_REGION', 'QRY_STRAND',
    #             'CI',
    #             'ALIGN_INDEX',
    #             'LEFT_SHIFT', 'HOM_REF', 'HOM_TIG',
    #             'CALL_SOURCE',
    #             'SEQ'
    #         ]
    #     )

    # Return tables
    return df_snv, df_insdel
