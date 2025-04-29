"""
Create and manaage alignemnt BED files.
"""

import numpy as np
import os
import pandas as pd

from .. import io

from . import features
from . import lcmodel
from . import op
from . import records
from . import score

# def get_align_bed(
#         align_filename: str,
#         df_qry_fai: pd.Series,
#         hap: str,
#         min_mapq: int=0,
#         score_model: score.ScoreModel | str=None,
#         lc_model: lcmodel.LCAlignModel=None,
#         align_features=None
# ):
#     """
#     Read alignment file as a BED file that PAV can process. Drops any records marked as unaligned by the SAM flag.
#
#     :param align_filename: Path to a SAM, CRAM, BAM, anything `pysam.AlignmentFile` can read.
#     :param df_qry_fai: Pandas Series with query names as keys and query lengths as values. Index should be cast as
#         type str if query names are numeric.
#     :param hap: Haplotype assinment for this alignment file (h1 or h2).
#     :param min_mapq: Minimum MAPQ. If 0, then all alignments are accepted as long as the unmapped flag is not set.
#     :param score_model: Alignment model object (`pavlib.align.score.ScoreModel`) or a configuration string to generate
#         a score model object. If `None`, the default score model is used. An alignment score is computed by summing
#         the score of each CIGAR operation against this model (match, mismatch, and gap) to create the "SCORE" column.
#     :param lc_model: Model for predicting low-confidence alignments.
#     :param align_features: List of alignment features (SCORE, MISMATCH_PROP, etc) to add to the alignment BED.
#
#     :return: BED file of alignment records.
#     """
#
#     if align_features is None:
#         align_features = features.ALIGN_FEATURE_COLUMNS
#     else:
#         align_features = list(align_features)
#
#     columns_head = [
#         '#CHROM', 'POS', 'END',
#         'INDEX',
#         'FILTER',
#         'QRY_ID', 'QRY_POS', 'QRY_END',
#         'QRY_ORDER',
#         'RG',
#         'MAPQ',
#         'IS_REV', 'FLAGS', 'HAP',
#         'CIGAR'
#     ]
#
#     columns = columns_head + align_features
#
#     if align_filename is None or os.stat(align_filename).st_size == 0:
#         return pd.DataFrame([], columns=columns)
#
#     # Get score model
#     score_model = score.get_score_model(score_model)
#
#     # Get records from SAM
#     record_list = list()
#     op_arr_list = list()
#
#     align_index = -1
#
#     with pysam.AlignmentFile(align_filename, 'rb') as in_file:
#         for record in in_file:
#
#             # Increment align_index
#             align_index += 1
#
#             # Skipped unmapped reads
#             if record.is_unmapped or record.mapping_quality < min_mapq or len(record.cigartuples) == 0:
#                 continue
#
#             # Get length for computing real query positions for rev-complemented records
#             qry_len = df_qry_fai[record.query_name]
#
#             # Read tags
#             tags = dict(record.get_tags())
#
#             # Determine left hard-clipped bases.
#             # pysam query alignment functions are relative to the sequence in the alignment record, not the original
#             # sequence. The left-most hard-clipped bases must be added to the query positions to translate to the
#             # correct query coordinates (https://github.com/pysam-developers/pysam/issues/1017).
#             if len(record.cigartuples) > 0:
#                 clip_h = record.cigartuples[0][1] if record.cigartuples[0][0] == op.H else 0
#             else:
#                 clip_h = 0
#
#             op_tuples = records.clip_op_tuples_soft_to_hard(record.cigartuples)
#
#             qry_map_pos = op_tuples[0][1] if op_tuples[0][0] == op.H else 0
#             qry_map_len = record.query_alignment_end - record.query_alignment_start
#             qry_map_end = qry_map_pos + qry_map_len
#
#             if record.query_alignment_start + clip_h != qry_map_pos:
#                 raise RuntimeError(f'First aligned based from pysam ({record.query_alignment_start}) does not match clipping ({qry_map_pos}) at alignment record {align_index}')
#
#             op_arr = np.array(op_tuples)
#             op_arr_list.append(op_arr)
#
#             # Disallow alignment match (M) in CIGAR (requires =X for base match/mismatch)
#             if op.M in {op_code for op_code, op_len in op_tuples}:
#                 raise RuntimeError((
#                     'Found alignment match CIGAR operation (M) for record {} (Start = {}:{}): '
#                     'Alignment requires CIGAR base-level match/mismatch (=X)'
#                 ).format(record.query_name, record.reference_name, record.reference_start))
#
#             # Save record
#             record_list.append(
#                 pd.Series(
#                     [
#                         record.reference_name,   # #CHROM
#                         record.reference_start,  # POS
#                         record.reference_end,    # END
#
#                         align_index,  # INDEX
#
#                         'PASS',  # FILTER
#
#                         record.query_name,  # QRY_ID
#                         qry_len - qry_map_end if record.is_reverse else qry_map_pos,  # QRY_POS
#                         qry_len - qry_map_pos if record.is_reverse else qry_map_end,  # QRY_END
#
#                         -1,  # QRY_ORDER (filled in later)
#
#                         tags['RG'] if 'RG' in tags else 'NA',  # RG
#                         tags['AO'] if 'AO' in tags else 'NA',  # AO
#
#                         record.mapping_quality,  # MAPQ
#
#                         record.is_reverse,       # IS_REV
#                         f'0x{record.flag:04x}',  # FLAGS
#                         hap,                     # HAP
#
#                         op.to_cigar_string(op_arr)  # CIGAR
#                     ],
#                     index=columns_head
#                 )
#             )
#
#     # Merge records
#     if len(record_list) == 0:
#         return pd.DataFrame(
#             [], columns=columns
#         )
#
#     df = pd.concat(record_list, axis=1).T
#
#     # Compute features
#     df = features.get_features(
#         df=df,
#         feature_list=align_features,
#         score_model=score_model,
#         existing_score_model=True,
#         op_arr_list=op_arr_list,
#         df_qry_fai=df_qry_fai,
#         inplace=True,
#         only_features=False,
#         force_all=True
#     )
#
#     # Set filter
#     if lc_model is not None:
#         df['FILTER'] = np.where(
#             lc_model(
#                 df,
#                 existing_score_model=score_model,
#                 op_arr_list=op_arr_list,
#                 qry_fai=df_qry_fai
#             ), records.FILTER_LCALIGN, 'PASS'
#         )
#
#     # Assign order per query sequence
#     df.sort_values(['QRY_ID', 'QRY_POS', 'QRY_END'], inplace=True)
#
#     for qry_id in df['QRY_ID'].unique():
#         df.loc[df['QRY_ID'] == qry_id, 'QRY_ORDER'] = df.loc[df['QRY_ID'] == qry_id, 'QRY_POS'].rank().astype(int) - 1
#
#     # Reference order
#     df.sort_values(['#CHROM', 'POS', 'END', 'QRY_ID'], ascending=[True, True, False, True], inplace=True)
#
#     # Check sanity
#     df.apply(records.check_record, df_qry_fai=df_qry_fai, axis=1)
#
#     # Check columns
#     missing_cols = [col for col in columns if col not in df.columns]
#
#     if missing_cols:
#         raise RuntimeError(f'Missing columns in alignment BED: {", ".join(missing_cols)}')
#
#     # Return BED
#     return df

def get_align_bed(
        align_filename: str,
        df_qry_fai: pd.Series,
        hap: str,
        min_mapq: int=0,
        score_model: score.ScoreModel | str=None,
        lc_model: lcmodel.LCAlignModel=None,
        align_features=None,
        check_match: bool=False,
        flag_filter=0x700,
        ref_fa_filename: str=None,
        qry_fa_filename: str=None
):
    """
    Read alignment records from a SAM file. Avoid pysam, it uses htslib, which has a limit of 268,435,456 bp for each
    alignment record, and clipping on a CIGAR string can exceed this limit
    (https://github.com/samtools/samtools/issues/1667) and causing PAV to crash with an error message starting with
    "CIGAR length too long at position".

    :param align_filename: File to read.
    :param df_qry_fai: Pandas Series with query names as keys and query lengths as values.
    :param hap: Name of haplotype.
    :param min_mapq: Minimum MAPQ score for alignment record.
    :param score_model: Score model to use.
    :param lc_model: LCAlignModel to use.
    :param align_features: List of alignment features (SCORE, MISMATCH_PROP, etc) to add to the alignment BED. If
        None, use PAV's default features.
    :param check_match: If `True`, check that all aligned bases in the final table either match (operation "=") or
        do not match (operation "X"). Time consuming and not required for production, only use for development and
        testing the SAM to BED conversion.
    :param flag_filter: Filter alignments matching these flags.
    :param ref_fa_filename: Reference FASTA filename. Required if `check_match` is True.
    :param qry_fa_filename: Query FASTA filename. Required if `check_match` is True.

    :return: Table of alignment records
    """

    if check_match:
        if ref_fa_filename is None:
            raise ValueError('ref_fa_filename is required if check_match is True')
        if qry_fa_filename is None:
            raise ValueError('qry_fa_filename is required if check_match is True')

    if align_features is None:
        align_features = features.ALIGN_FEATURE_COLUMNS
    else:
        align_features = list(align_features)

    columns_head = [
        '#CHROM', 'POS', 'END',
        'INDEX',
        'FILTER',
        'QRY_ID', 'QRY_POS', 'QRY_END',
        'QRY_ORDER',
        'RG',
        'MAPQ',
        'IS_REV', 'FLAGS', 'HAP',
        'CIGAR'
    ]

    columns = columns_head + align_features

    if align_filename is None or os.stat(align_filename).st_size == 0:
        return pd.DataFrame([], columns=columns)

    # Get score model
    score_model = score.get_score_model(score_model)

    # Get records from SAM
    record_list = list()
    op_arr_list = list()

    align_index = -1
    line_number = 0

    with io.SamStreamer(align_filename) as in_file:
        for line in in_file:
            line_number += 1

            try:

                line = line.strip()

                if line.startswith('@'):
                    continue

                align_index += 1

                tok = line.split('\t')

                if len(tok) < 11:
                    raise RuntimeError('Expected at least 11 fields, received {}'.format(line_number, len(tok)))

                tag = dict(val.split(':', 1) for val in tok[11:])  # Note: values are prefixed with type and colon, (e.g. {"NM": "i:579204"}).

                if 'CG' in tag:
                    raise RuntimeError(f'Found BAM-only CG')

                if 'RG' in tag:
                    if not tag['RG'].startswith('Z:'):
                        raise RuntimeError(f'Found non-Z RG tag: {tag["RG"]}')
                    tag_rg = tag['RG'][2:].strip()

                    if not tag_rg:
                        tag_rg = 'NA'

                else:
                    tag_rg = 'NA'

                flag = int(tok[1])
                mapq = int(tok[4])
                is_rev = bool(flag & 0x10)

                pos_ref = int(tok[3]) - 1

                # Skipped unmapped reads, low MAPQ reads, or other flag-based filters
                if flag & 0x4 or mapq < min_mapq or pos_ref < 0:
                    continue

                # Get alignment operations
                op_arr = op.clip_soft_to_hard(op.cigar_as_array(tok[5]))

                if np.any(op_arr[:, 0] * op.M):
                    raise RuntimeError('PAV does not allow match alignment operations (CIGAR "M", requires "=" and "X")')

                len_qry = np.sum(op_arr[np.isin(op_arr[:, 0], op.CONSUMES_QRY_ARR), 1])
                len_ref = np.sum(op_arr[np.isin(op_arr[:, 0], op.CONSUMES_REF_ARR), 1])

                if is_rev:
                    pos_qry = op_arr[-1, 1] * (op_arr[-1, 0] == op.H)
                else:
                    pos_qry = op_arr[0, 1] * (op_arr[0, 0] == op.H)

                # Check sequences
                chrom = tok[2].strip()
                qry_id = tok[0].strip()

                if chrom == '*' or qry_id == '*':
                    raise RuntimeError(f'Found mapped read with missing names (chrom={chrom}, qry_id={qry_id})')

                # Save record
                record_list.append(
                    pd.Series(
                        [
                            chrom,              # #CHROM
                            pos_ref,            # POS
                            pos_ref + len_ref,  # END

                            align_index,  # INDEX

                            'PASS' if not flag & flag_filter or mapq < min_mapq else records.FILTER_ALIGN,  # FILTER

                            qry_id,             # QRY_ID
                            pos_qry,            # QRY_POS
                            pos_qry + len_qry,  # QRY_END

                            -1,  # QRY_ORDER (filled in later)

                            tag_rg,  # RG

                            mapq,  # MAPQ

                            is_rev,           # IS_REV
                            f'0x{flag:04x}',  # FLAGS
                            hap,              # HAP

                            op.to_cigar_string(op_arr)  # CIGAR
                        ],
                        index=columns_head
                    )
                )

                op_arr_list.append(op_arr)

            except Exception as e:
                raise RuntimeError('Failed to parse record at line {}: {}'.format(line_number, str(e))) from e

    # Merge records
    if len(record_list) == 0:
        return pd.DataFrame(
            [], columns=columns
        )

    df = pd.concat(record_list, axis=1).T
    df['QRY_ORDER'] = get_qry_order(df)

    # Compute features
    df = features.get_features(
        df=df,
        feature_list=align_features,
        score_model=score_model,
        existing_score_model=True,
        op_arr_list=op_arr_list,
        df_qry_fai=df_qry_fai,
        inplace=True,
        only_features=False,
        force_all=True
    )

    # Set filter
    if lc_model is not None:
        filter_loc =  lc_model(
            df,
            existing_score_model=score_model,
            op_arr_list=op_arr_list,
            qry_fai=df_qry_fai
        )

        df.loc[filter_loc, 'FILTER'] = df.loc[filter_loc, 'FILTER'].apply(lambda vals:
            ','.join(sorted(set(vals.split(',')) - {'PASS'} | {records.FILTER_LCALIGN}))
        )

    # Reference order
    df.sort_values(['#CHROM', 'POS', 'END', 'QRY_ID'], ascending=[True, True, False, True], inplace=True)

    # Check sanity
    df.apply(records.check_record, df_qry_fai=df_qry_fai, axis=1)

    # Check columns
    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        raise RuntimeError(f'Missing columns in alignment BED: {", ".join(missing_cols)}')

    # Check alignment sanity by comparing matched bases
    if check_match:
        df.apply(records.check_matched_bases, ref_fa_filename=ref_fa_filename, qry_fa_filename=qry_fa_filename, axis=1)

    # Return BED
    return df

def aggregate_alignment_records(
        df_align: pd.DataFrame,
        df_qry_fai: pd.Series,
        score_model: bool=None,
        min_score: float=None,
        noncolinear_penalty: bool=True
):
    """
    Aggregate colinear alignment records.

    :param df_align: Table of alignment records. MUST be query trimmed (or query- & reference-trimmed)
    :param df_qry_fai: Query FAI.
    :param score_model: Model for scoring INS and DEL between alignment records. If none, use the default model.
    :param min_score: Do not aggregate alignment records with a score below this value. Defaults to the score of a
        10 kbp gap.
    :param noncolinear_penalty: When aggregating two records, add a gap penalty equal to the difference between the
        unaligned reference and query bases between the records. This penalizes non-colinear alignments.

    :return: Table of aggregated alignment records.
    """

    # Check parameters
    if score_model is None:
        score_model = score.get_score_model()

    if min_score is None:
        min_score = score_model.gap(10000)

    min_agg_index = int(10 ** np.ceil(np.log10(
        np.max(df_align['INDEX'])
    ))) - 1  # Start index for aggregated records at the next power of 10

    next_agg_index = min_agg_index + 1

    # Sort
    df_align = df_align.sort_values(['QRY_ID', 'QRY_POS']).copy()
    df_align['INDEX_PREAGG'] = df_align['INDEX']

    # Return existing table if empty
    if df_align.shape[0] == 0:
        df_align['INDEX_PREAGG'] = df_align['INDEX']
        return df_align

    df_align['INDEX_PREAGG'] = df_align['INDEX'].apply(lambda val: [val])
    df_align['MAPQ'] = df_align['MAPQ'].apply(lambda val: [val])
    df_align['FLAGS'] = df_align['FLAGS'].apply(lambda val: [val])

    # Find and aggregate near co-linear records over SVs
    align_records = list()  # Records that were included in a merge

    for qry_id in sorted(set(df_align['QRY_ID'])):
        df = df_align.loc[df_align['QRY_ID'] == qry_id]
        i_max = df.shape[0] - 1

        i = 0
        row1 = df.iloc[i]

        while i < i_max:
            i += 1

            row2 = row1
            row1 = df.iloc[i]

            # Skip if chrom or orientation is not the same
            if row1['#CHROM'] != row2['#CHROM'] or row1['IS_REV'] != row2['IS_REV']:
                align_records.append(row2)
                continue

            # Get reference distance
            if row1['IS_REV']:
                ref_dist = row2['POS'] - row1['END']
            else:
                ref_dist = row1['POS'] - row2['END']

            qry_dist = row1['QRY_POS'] - row2['QRY_END']

            if qry_dist < 0:
                raise RuntimeError(f'Query distance is negative: {qry_dist}: alignment indexes {row1["INDEX"]} and {row2["INDEX"]}')

            if ref_dist >= 0:
                # Contiguous in reference space, check query space

                # Score gap between the alignment records
                this_score = score_model.gap(ref_dist) + score_model.gap(qry_dist)

                score_gap = this_score + (
                    score_model.gap(np.abs(qry_dist - ref_dist)) if noncolinear_penalty else 0
                )

                if score_gap < min_score:
                    align_records.append(row2)
                    continue

                #
                # Aggregate
                #
                row1 = row1.copy()

                # Set query position
                row1['QRY_POS'] = row2['QRY_POS']

                # Get rows in order
                if row1['IS_REV']:
                    row_l = row1
                    row_r = row2

                    row1['END'] = row2['END']

                    row1['TRIM_REF_R'] = row2['TRIM_REF_R']
                    row1['TRIM_QRY_R'] = row2['TRIM_QRY_R']

                else:
                    row_l = row2
                    row_r = row1

                    row1['POS'] = row2['POS']

                    row1['TRIM_REF_L'] = row2['TRIM_REF_L']
                    row1['TRIM_QRY_L'] = row2['TRIM_QRY_L']

                # Set records
                row1['FLAGS'] = row1['FLAGS'] + row2['FLAGS']
                row1['MAPQ'] = row1['MAPQ'] + row2['MAPQ']
                row1['INDEX_PREAGG'] = row1['INDEX_PREAGG'] + row2['INDEX_PREAGG']
                row1['SCORE'] = row1['SCORE'] + row2['SCORE']

                if 'RG' in row1:
                    row1['RG'] = np.nan

                if 'AO' in row1:
                    row1['AO'] = np.nan

                # Merge CIGAR strings
                op_arr_l = op.cigar_as_array(row_l['CIGAR'])
                op_arr_r = op.cigar_as_array(row_r['CIGAR'])

                while op_arr_l.shape[0] > 0 and op_arr_l[-1, 0] in op.CLIP_SET:  # Tail of left record
                    op_arr_l = op_arr_l[:-1]

                while op_arr_r.shape[0] > 0 and op_arr_r[0, 0] in op.CLIP_SET:  # Head of right record
                    op_arr_r = op_arr_r[1:]

                ins_len = qry_dist
                del_len = ref_dist

                if qry_dist > 0:
                    while op_arr_l.shape[0] > 0 and op_arr_l[-1, 0] == op.I:  # Concat insertions (no "...xIxI..." in CIGAR)
                        ins_len += op_arr_l[-1, 1]
                        op_arr_l = op_arr_l[:-1]

                    op_arr_l = np.append(op_arr_l, np.array([[ins_len, op.I]]), axis=0)

                if ref_dist > 0:
                    while op_arr_l.shape[0] > 0 and op_arr_l[-1, 0] == op.D:  # Concat deletions (no "...xDxD..." in CIGAR)
                        del_len += op_arr_l[-1, 1]
                        op_arr_l = op_arr_l[:-1]

                    op_arr_l = np.append(op_arr_l, np.array([[del_len, op.D]]), axis=0)

                op_arr = np.append(op_arr_l, op_arr_r, axis=0)

                row1['CIGAR'] = op.to_cigar_string(op_arr)
                row1['SCORE'] = score_model.score_operations(op_arr)

                # Set alignment indexes
                if row2['INDEX'] < min_agg_index:
                    # Use the next aggregate index
                    row1['INDEX'] = next_agg_index
                    next_agg_index += 1

                else:
                    # row2 was aggregated, use its aggregate index
                    row1['INDEX'] = row2['INDEX']

                # Check
                records.check_record(row1, df_qry_fai)

            else:
                align_records.append(row2)

        # Add last record
        align_records.append(row1)

    # Concatenate records
    df = pd.concat(align_records, axis=1).T

    df['MAPQ'] = df['MAPQ'].apply(lambda val: ','.join([str(v) for v in val]))
    df['FLAGS'] = df['FLAGS'].apply(lambda val: ','.join([str(v) for v in val]))
    df['INDEX_PREAGG'] = df['INDEX_PREAGG'].apply(lambda val: ','.join([str(v) for v in val]))

    # Assign order per query sequence
    df['QRY_ORDER'] = get_qry_order(df)

    # Reference order
    df.sort_values(['#CHROM', 'POS', 'END', 'QRY_ID'], ascending=[True, True, False, True], inplace=True)

    # Check sanity (whole table, modified records already checked, should pass)
    df.apply(records.check_record, df_qry_fai=df_qry_fai, axis=1)

    return df

def align_bed_to_depth_bed(df, df_fai=None, index_sep=',', retain_filtered=False):
    """
    Get a BED file of alignment depth from an alignment BED.

    Output columns:
    * #CHROM: Reference chromosome.
    * POS: Reference start position (BED coordinates).
    * END: Reference end position (BED coordinates).
    * DEPTH: Number of alignment records. Integer, may be 0.
    * QUERY: Query names. Comma-separated list if multiple queries.
    * INDEX: Query indexes in the same order as QUERIES (corresponding INDEX column in df).

    If `df_fai` is not 'None', then depth records extend from 0 to the end of the chromosome. If the first record does
    not reach position 0, a 0-depth record is added over that region. Similarly, if the last record does not reach the
    end of the chromosome, a 0-depth record is added over that region. If `df_fai` is `None`, then no padding is done
    to the start or end of the chromosome and 0-depth records only appear between alignment records.

    :param df: Alignment BED file.
    :param df_fai: FAI series (keys = reference sequence names, values = sequence lengths).
    :param index_sep: Entries in the index are separated by this character
    :param retain_filtered: Do not drop filtered alignment records if True.

    :return: A Pandas DataTable with depth across all reference loci.
    """

    # Get a list of alignment events (start and end alignment record)

    # Build an ordered list of alignments.
    # align_list is a list of tuples:
    #  0) Chromosome
    #  1) Event position
    #  2) Event type (1 is align start, 0 is align end)
    #  3) Align record index. Removes the correct alignment from the aligned query list when an alignment record ends.
    #  4) Query ID: List of query IDs in the aligned region (comma-separated list). Each alignment start goes to the
    #     end of the list, and each alignment end removes the element from the list that it added even if the query
    #     ID is the same.

    align_list = list()

    if not retain_filtered and 'FILTER' in df.columns:
        is_filtered = (df['FILTER'] != 'PASS') & ~ pd.isnull(df['FILTER'])
    else:
        is_filtered = pd.Series(False, index=df.index)

    for index, row in df.iterrows():

        if is_filtered[index]:
            continue

        align_list.append(
            (str(row['#CHROM']), row['POS'], 1, row['INDEX'], row['QRY_ID'], row['INDEX'])
        )

        align_list.append(
            (str(row['#CHROM']), row['END'], 0, row['INDEX'], row['QRY_ID'], row['INDEX'])
        )

    if not align_list:
        raise RuntimeError('No alignments to process')

    align_list = sorted(align_list)

    if align_list[0][2] != 1:
        raise RuntimeError(f'First alignment is not a record start: {",".join([str(val) for val in align_list[0]])}')

    # Setup to process BED records
    df_bed_list = list()

    last_chrom = None
    last_pos = None
    qry_list = list()

    # Write empty records for reference sequences with no alignments
    if df_fai is not None:
        seq_fai_list = list(sorted(df_fai.index.astype(str)))

        while len(seq_fai_list) > 0 and seq_fai_list[0] < align_list[0][0]:
            df_bed_list.append(
                pd.Series(
                    [seq_fai_list[0], 0, df_fai[seq_fai_list[0]], 0, '', ''],
                    index=['#CHROM', 'POS', 'END', 'DEPTH', 'QRY_ID', 'INDEX']
                )
            )

            seq_fai_list = seq_fai_list[1:]

        if len(seq_fai_list) == 0 or seq_fai_list[0] != align_list[0][0]:
            raise RuntimeError(f'Missing {align_list[0][0]} in FAI or out of order')

    else:
        seq_fai_list = None

    # Process BED records
    for chrom, pos, event, index, qry, row_index in align_list:
        # print(f'{chrom} {pos} {event} {index} {qry} {row_index}')  # DBGTMP

        # Chromosome change
        if chrom != last_chrom:

            # Check sanity
            if qry_list:
                raise RuntimeError(f'Switched chromosome ({last_chrom} > {chrom}) with open queries: {", ".join(qry_list)}')

            # Check chromosome order
            if df_fai is not None:

                # Check chromosome in FAI
                if chrom not in df_fai.index:
                    raise RuntimeError(f'Missing chromosome in reference FAI index: {chrom}')

                # Write empty records for reference sequences with no alignments
                while len(seq_fai_list) > 0 and seq_fai_list[0] < chrom:
                    df_bed_list.append(
                        pd.Series(
                            [seq_fai_list[0], 0, df_fai[seq_fai_list[0]], 0, '', ''],
                            index=['#CHROM', 'POS', 'END', 'DEPTH', 'QRY_ID', 'INDEX']
                        )
                    )

                    seq_fai_list = seq_fai_list[1:]

                if len(seq_fai_list) == 0 or seq_fai_list[0] != chrom:
                    raise RuntimeError(f'Missing {chrom} in FAI or out of order')

                seq_fai_list = seq_fai_list[1:]

                # Add record up to end of chromosome
                if last_chrom is not None:
                    if last_pos > df_fai[last_chrom]:
                        raise RuntimeError(f'Last END position for chromosome {last_chrom} is greater than chromosome length: {last_pos} > {df_fai[last_chrom]}')

                    if last_pos < df_fai[last_chrom]:
                        df_bed_list.append(
                            pd.Series(
                                [
                                    last_chrom, last_pos, df_fai[last_chrom], len(qry_list),
                                    ','.join([val[1] for val in qry_list]),
                                    index_sep.join([str(val[0]) for val in qry_list])
                                ],
                                index=['#CHROM', 'POS', 'END', 'DEPTH', 'QRY_ID', 'INDEX']
                            )
                        )

            # Add chrom:0-pos record
            if df_fai is not None and pos > 0:
                df_bed_list.append(
                    pd.Series(
                        [
                            chrom, 0, pos, len(qry_list),
                            ','.join([val[1] for val in qry_list]),
                            index_sep.join([str(val[0]) for val in qry_list])
                        ],
                        index=['#CHROM', 'POS', 'END', 'DEPTH', 'QRY_ID', 'INDEX']
                    )
                )

            # Set state for the next alignment
            last_chrom = chrom
            last_pos = pos

        # Check position sanity
        if last_pos > pos:
            raise RuntimeError(f'Position out of order: {last_chrom}:{last_pos} > {chrom}:{pos}')

        # Write last record
        if pos > last_pos:
            df_bed_list.append(
                pd.Series(
                    [
                        chrom, last_pos, pos, len(qry_list),
                        ','.join([val[1] for val in qry_list]),
                        index_sep.join([str(val[0]) for val in qry_list])
                    ],
                    index=['#CHROM', 'POS', 'END', 'DEPTH', 'QRY_ID', 'INDEX']
                )
            )

            last_pos = pos

        # Process event
        if event == 1:
            qry_list.append((index, qry))

        elif event == 0:
            n = len(qry_list)

            if n == 0:
                raise RuntimeError(f'Got END event with no queries in the list: {chrom},{pos},{event},{index},{qry}')

            qry_list = [val for val in qry_list if val != (index, qry)]

            if len(qry_list) == n:
                raise RuntimeError(f'Could not find query to END in query list: chrom={chrom},pos={pos},event={event},index="{index}",qry={qry}')

            if len(qry_list) < n - 1:
                raise RuntimeError(f'END removed multiple queries: chrom={chrom},pos={pos},event={event},index="{index}",qry={qry}')

        else:
            raise RuntimeError(f'Unknown event type event={event}: chrom={chrom},pos={pos},event={event},index="{index}",qry={qry}')

    # Check final state
    if qry_list:
        raise RuntimeError(f'Ended alignment records with open queries: {", ".join(qry_list)}')

    if df_fai is not None:
        if last_chrom not in df_fai.index:
            raise RuntimeError(f'Missing chromosome in reference FAI index: {last_chrom}')

        # Add final record up to end of chromosome
        if last_pos > df_fai[last_chrom]:
            raise RuntimeError(f'Last END position for chromosome {last_chrom} is greater than chromosome length: {last_pos} > {df_fai[last_chrom]}')

        if last_pos < df_fai[last_chrom]:
            df_bed_list.append(
                pd.Series(
                    [
                        last_chrom, last_pos, df_fai[last_chrom], len(qry_list),
                        ','.join([val[1] for val in qry_list]),
                        index_sep.join([str(val[0]) for val in qry_list])
                    ],
                    index=['#CHROM', 'POS', 'END', 'DEPTH', 'QRY_ID', 'INDEX']
                )
            )

        # Write empty records for reference sequences with no alignments
        while len(seq_fai_list) > 0:

            df_bed_list.append(
                pd.Series(
                    [seq_fai_list[0], 0, df_fai[seq_fai_list[0]], 0, '', ''],
                    index=['#CHROM', 'POS', 'END', 'DEPTH', 'QRY_ID', 'INDEX']
                )
            )

            seq_fai_list = seq_fai_list[1:]

    # Create BED file
    df_bed = pd.concat(df_bed_list, axis=1).T

    df_bed.sort_values(['#CHROM', 'POS'], inplace=True)

    return df_bed

def get_qry_order(df):
    """
    Get a column describing the query order of each alignment record. For any query sequence, the first alignment
    record in the sequence (i.e. containing the left-most aligned base relative to the query sequence) will have
    order 0, the next alignment record 1, etc. The order is set per query sequence (i.e. the first aligned record
    of every unique query ID will have order 0).

    :param df: DataFrame of alignment records.

    :return: A Series of alignment record query orders.
    """

    df_qry_id = df.sort_values(['QRY_ID', 'QRY_POS', 'QRY_END'])['QRY_ID']
    df_qry_ord = pd.Series(range(df_qry_id.shape[0]), index=df_qry_id.index)

    min_ord = pd.concat([df_qry_id, df_qry_ord], axis=1).groupby('QRY_ID').min()[0]

    row_ord = pd.Series((df_qry_ord - df_qry_id.apply(min_ord.get)).reindex(df.index))
    row_ord.name = 'QRY_ORDER'

    return row_ord
