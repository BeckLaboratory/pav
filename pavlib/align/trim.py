# Alignment trimming functions

import numpy as np
import pandas as pd

import svpoplib

from .. import pavconfig

from . import features
from . import op
from . import records
from . import score
from . import tables

# Indices for tuples returned by trace_op_to_zero()
TC_INDEX = 0
TC_OP_CODE = 1
TC_OP_LEN = 2
TC_DIFF_CUM = 3
TC_DIFF = 4
TC_EVENT_CUM = 5
TC_EVENT = 6
TC_REF_BP = 7
TC_QRY_BP = 8
TC_CLIPS_BP = 9
TC_CLIPH_BP = 10
TC_SCORE_CUM = 11

TRIM_DESC = {
    'none': 'No trimming',
    'qry': 'Query-only trimming',
    'qryref': 'Query-Reference trimming'
}

def trim_alignments(
        df: pd.DataFrame,
        df_qry_fai: pd.Series,
        min_trim_qry_len: int=1,
        match_qry: bool=False,
        mode: str='both',
        score_model: score.ScoreModel=None,
        pav_params: pavconfig.ConfigParams=None
):
    """
    Do alignment trimming from prepared alignment BED file. This BED contains information about reference and
    query coordinates mapped, CIGAR string and flags.

    :param df: Alignment dataframe.
    :param min_trim_qry_len: Minimum alignment record length. Alignment records smaller will be discarded.
    :param df_qry_fai: FAI file name from the query FASTA that was mapped, or a FAI series already read to memory
        (pd.Series). Used for an alignment sanity check after trimming.
    :param match_qry: When trimming in reference space (removing redundantly mapped reference bases), only trim
        alignment records if the query ID matches. This allows multiple queries to map to the same location,
        which may produce a redundant set of variant calls from mis-aligned queries (e.g. incorrectly mapped paralogs).
        This mode is useful for generating a maximal callset where multiple queries may cover the same region (e.g.
        alleles in squashed assembly or mixed haplotypes (e.g. subclonal events). Additional QC should be applied to
        callsets using this option.
    :param mode: Trim alignments to remove redundantly mapped query bases (mode = "qry"), reference bases with more than
        one alignment (mode = "ref"), or both (mode = "both"). If None, assume "both".
    :param score_model: Alignment model object (`pavlib.align.score.ScoreModel`) or a configuration string to generate
        a score model object. If `None`, the default score model is used. An alignment score is computed by summing
        the score of each CIGAR operation against this model (match, mismatch, and gap) to update the "SCORE" column.
    :param pav_params: PAV configuration parameters. If None, a default configuration is used.

    :return: Trimmed alignments as an alignment DataFrame. Same format as `df` with columns added describing the
        number of reference and query bases that were trimmed. Dropped records (mapped inside another or too shart) are
        removed.
    """

    if pav_params is None:
        pav_params = pavconfig.ConfigParams()

    if score_model is None:
        score_model = score.get_score_model(pav_params.align_score_model)
    else:
        score.get_score_model(score_model)  # Convert to object if score_model is a string

    # Check mode
    if mode is None:
        mode = 'both'

    mode = mode.strip().lower()

    if mode == 'qry':
        do_trim_qry = True
        do_trim_ref = False
    elif mode == 'ref':
        do_trim_qry = False
        do_trim_ref = True
    elif mode == 'both':
        do_trim_qry = True
        do_trim_ref = True
    else:
        raise RuntimeError(f'Unrecognized trimming mode "{mode}": Expected "qry", "ref", or "both"')

    # For each altered alignment record, this dict contains the operation array (op_code: first column,
    # op_len: second column) and is keyed by the INDEX value of the alignment record since the indexes may change
    # and the INDEX column is unique.
    op_arr_dict = dict()

    # Remove short alignments
    df = df.loc[df['QRY_END'] - df['QRY_POS'] >= min_trim_qry_len].copy()


    ###                     ###
    ### Trim in query space ###
    ###                     ###

    if do_trim_qry:

        # Sort by alignment lengths in query space
        df = df.sort_values(['QRY_ID', 'SCORE'], ascending=(True, False)).reset_index(drop=True)

        # Do trim in query space #
        iter_index_l = 0
        index_max = df.shape[0]

        while iter_index_l < index_max:

            if df.loc[iter_index_l, 'INDEX'] < 0:
                iter_index_l += 1
                continue

            iter_index_r = iter_index_l + 1

            while iter_index_r < index_max and df.loc[iter_index_l, 'QRY_ID'] == df.loc[iter_index_r, 'QRY_ID']:

                if df.loc[iter_index_l, 'INDEX'] < 0:
                    iter_index_l += 1
                    continue

                # Get index in order of query placement
                index_l, index_r = (iter_index_l, iter_index_r) \
                    if df.loc[iter_index_l, 'QRY_POS'] <= df.loc[iter_index_r, 'QRY_POS'] \
                        else (iter_index_r, iter_index_l)

                row_l = df.loc[index_l]
                row_r = df.loc[index_r]

                # # Get index in order of query placement
                # if df.loc[iter_index_l, 'QRY_POS'] <= df.loc[iter_index_r, 'QRY_POS']:
                #     index_l = iter_index_l
                #     index_r = iter_index_r
                # else:
                #     index_l = iter_index_r
                #     index_r = iter_index_l

                # # Skip if one record was already removed
                # if row_l['INDEX'] < 0 or row_r['INDEX'] < 0:
                #     iter_index_r += 1
                #     continue

                # Skip if there is no overlap
                if row_r['QRY_POS'] >= row_l['QRY_END']:
                    iter_index_r += 1
                    continue

                # # Check for record fully contained within another
                # if row_r['QRY_END'] <= row_l['QRY_END']:
                #     df.loc[index_r, 'INDEX'] = -1
                #     iter_index_r += 1
                #     continue

                # Determine trim orientation (right side of index_l is to be trimmed, must be reversed so
                # trimmed alignment records are at the beginning; left side of index_r is to be trimmed, which is
                # already at the start of the alignment operation list).
                rev_l = not row_l['IS_REV']  # Trim right end of index_l
                rev_r = row_r['IS_REV']      # Trim left end of index_r

                # Get operation arrays
                op_arr_l = get_op_arr_from_dict(row_l, op_arr_dict)
                op_arr_r = get_op_arr_from_dict(row_r, op_arr_dict)

                # Determine which side to preferentially trim in case of ties
                if (
                    rev_l != rev_r  # Same as row_l['IS_REV'] == row_r['IS_REV']
                ) and (
                    row_l['#CHROM'] == row_r['#CHROM']  # Same chromosome
                ):
                    # Same chromosome and orientation, preferentially trim left-most record to maintain left-aligning
                    # breakpoints.

                    prefer_l = (
                        row_l['QRY_POS'] if row_l['IS_REV'] else row_l['QRY_END']  # Reference position on left record being trimmed
                    ) <= (
                        row_r['QRY_END'] if row_r['IS_REV'] else row_r['QRY_POS']  # Reference position on right record being trimmed
                    )

                else:
                    prefer_l = row_l['SCORE'] <= row_r['SCORE']

                if prefer_l:
                    record_l, record_r, op_arr_l, op_arr_r = trim_alignment_record(
                        df.loc[index_l],
                        df.loc[index_r],
                        'qry',
                        rev_l=rev_l,
                        rev_r=rev_r,
                        score_model=score_model,
                        op_arr_l=op_arr_l,
                        op_arr_r=op_arr_r
                    )
                else:
                    record_r, record_l, op_arr_r, op_arr_l = trim_alignment_record(
                        df.loc[index_r],
                        df.loc[index_l],
                        'qry',
                        rev_l=rev_r,
                        rev_r=rev_l,
                        score_model=score_model,
                        op_arr_l=op_arr_r,
                        op_arr_r=op_arr_l
                    )

                # # If query alignments overlap in reference space and are in the same orientation, preferentially
                # # trim the downstream end more to left-align alignment-truncating SVs (first argument to
                # # trim_alignment_record is preferentially trimmed)
                # if ref_overlap:
                #
                #     # Overlap in reference space (same orientation)
                #     # Try both trim orientations and choose the one that left-aligns the best
                #
                #     # a: Try with record l as first and r as second
                #     record_l_a, record_r_a, op_arr_l_a, op_arr_r_a = trim_alignment_record(
                #         df.loc[index_l],
                #         df.loc[index_r],
                #         'qry',
                #         rev_l=rev_l,
                #         rev_r=rev_r,
                #         score_model=score_model,
                #         op_arr_l=op_arr_l,
                #         op_arr_r=op_arr_r
                #     )
                #
                #     # b: Try with record r as first and l as second
                #     record_l_b, record_r_b, op_arr_l_b, op_arr_r_b = trim_alignment_record(
                #         df.loc[index_r],
                #         df.loc[index_l],
                #         'qry',
                #         rev_l=rev_r,
                #         rev_r=rev_l,
                #         score_model=score_model,
                #         op_arr_l=op_arr_r,
                #         op_arr_r=op_arr_l
                #     )
                #
                #     ### Determine which left-aligns best ###
                #     keep = None
                #
                #     # Case: Alignment trimming completely removes one of the records
                #     rm_l_a = record_l_a['QRY_END'] - record_l_a['QRY_POS'] < min_trim_qry_len
                #     rm_l_b = record_l_b['QRY_END'] - record_l_b['QRY_POS'] < min_trim_qry_len
                #
                #     rm_r_a = record_r_a['QRY_END'] - record_r_a['QRY_POS'] < min_trim_qry_len
                #     rm_r_b = record_r_b['QRY_END'] - record_r_b['QRY_POS'] < min_trim_qry_len
                #
                #     rm_any_a = rm_l_a or rm_r_a
                #     rm_any_b = rm_l_b or rm_r_b
                #
                #     # Break tie if one way removes a record and the other does not
                #     if rm_any_a and not rm_any_b:
                #         if not rm_l_a and rm_r_a:
                #             keep = 'a'
                #
                #     elif rm_any_b and not rm_any_a:
                #         if not rm_l_b and rm_r_b:
                #             keep = 'b'
                #
                #     # Break tie if both are removed in one (do not leave short alignments).
                #     if keep is None and rm_any_a:  # Both l and r are None, case where one is None was checked
                #         keep = 'a'
                #
                #     if keep is None and rm_any_b:  # Both l and r are None, case where one is None was checked
                #         keep = 'b'
                #
                #     # Break tie on most left-aligned base
                #     if keep is None:
                #
                #         # Get position at end of trim
                #         trim_pos_l_a = record_l_a['END'] if not record_l_a['IS_REV'] else record_l_a['POS']
                #         trim_pos_l_b = record_l_b['END'] if not record_l_b['IS_REV'] else record_l_b['POS']
                #
                #         if trim_pos_l_a <= trim_pos_l_b:
                #             keep = 'a'
                #         else:
                #             keep = 'b'
                #
                #     # Set record_l and record_r to the kept record
                #     if keep == 'a':
                #         record_l = record_l_a
                #         record_r = record_r_a
                #         op_arr_l = op_arr_l_a
                #         op_arr_r = op_arr_r_a
                #
                #     else:
                #         # Note: record at index_l became record_r_b (index_r become record_l_b)
                #         # Swap back to match the index
                #         record_l = record_r_b
                #         record_r = record_l_b
                #         op_arr_l = op_arr_r_b
                #         op_arr_r = op_arr_l_b
                #
                # else:
                #     # Does not overlap in reference space in same orientation
                #
                #     # Switch record order if they are on the same query and same orientation (note: rev_l and rev_r are
                #     # opposite if query sequences are mapped in the same orientation, one was swapped to trim the
                #     # downstream-aligned end).
                #     if row_l['#CHROM'] == row_r['#CHROM'] and rev_l != rev_r:
                #
                #         # Get position of end to be trimmed
                #         trim_pos_l = row_l['END'] if not row_l['IS_REV'] else row_l['POS']
                #         trim_pos_r = row_r['POS'] if not row_r['IS_REV'] else row_r['END']
                #
                #         # Swap positions so the upstream-aligned end of the query is index_l. The left end is
                #         # preferentially trimmed shorter where there are equal breakpoints effectively left-aligning
                #         # around large SVs (e.g. large DELs).
                #         if trim_pos_r < trim_pos_l:
                #
                #             # Swap
                #             row_l, row_r = row_r, row_l
                #             rev_l, rev_r = rev_r, rev_l
                #             index_l, index_r = index_r, index_l
                #             op_arr_l, op_arr_r = op_arr_r, op_arr_l
                #
                #     # Trim record
                #     record_l, record_r, op_arr_l, op_arr_r = trim_alignment_record(
                #         row_l, row_r, 'qry',
                #         rev_l=rev_l,
                #         rev_r=rev_r,
                #         score_model=score_model,
                #         op_arr_l=op_arr_l,
                #         op_arr_r=op_arr_r
                #     )

                # Modify if new aligned size is at least min_trim_qry_len, remove if shorter
                if record_l['QRY_END'] - record_l['QRY_POS'] >= min_trim_qry_len:
                    df.loc[index_l] = record_l
                    op_arr_dict[record_l['INDEX']] = op_arr_l
                else:
                    df.loc[index_l, 'INDEX'] = -1

                if (record_r['QRY_END'] - record_r['QRY_POS']) >= min_trim_qry_len:
                    df.loc[index_r] = record_r
                    op_arr_dict[record_r['INDEX']] = op_arr_r

                else:
                    df.loc[index_r, 'INDEX'] = -1

                # Next r record
                iter_index_r += 1

            # Next l record
            iter_index_l += 1

        # Discard fully trimmed records
        df = df.loc[df['INDEX'] >= 0].copy()


    ###                         ###
    ### Trim in reference space ###
    ###                         ###

    if do_trim_ref:

        # Sort by alignment length in reference space
        df = df.sort_values(['#CHROM', 'SCORE'], ascending=(True, False)).reset_index(drop=True)

        # df = df.loc[
        #     pd.concat(
        #         [df['#CHROM'], df['END'] - df['POS']], axis=1
        #     ).sort_values(
        #         ['#CHROM', 0],
        #         ascending=(True, False)
        #     ).index
        # ].reset_index(drop=True)

        # Do trim in reference space
        iter_index_l = 0
        index_max = df.shape[0]

        while iter_index_l < index_max:
            iter_index_r = iter_index_l + 1

            while (
                    iter_index_r < index_max and
                    df.loc[iter_index_l, '#CHROM'] == df.loc[iter_index_r, '#CHROM']
            ):

                # Skip if one record was already removed
                if df.loc[iter_index_l, 'INDEX'] < 0 or df.loc[iter_index_r, 'INDEX'] < 0:
                    iter_index_r += 1
                    continue

                # Skip if match_qry and query names differ
                if match_qry and df.loc[iter_index_l, 'QRY_ID'] != df.loc[iter_index_r, 'QRY_ID']:
                    iter_index_r += 1
                    continue

                # Get indices ordered by query placement
                index_l, index_r = (iter_index_l, iter_index_r) \
                    if df.loc[iter_index_l, 'POS'] <= df.loc[iter_index_r, 'POS'] \
                    else (iter_index_r, iter_index_l)

                row_l = df.loc[index_l]
                row_r = df.loc[index_r]

                op_arr_l = get_op_arr_from_dict(row_l, op_arr_dict)
                op_arr_r = get_op_arr_from_dict(row_r, op_arr_dict)

                # # Get indices ordered by query placement
                # if df.loc[iter_index_l, 'POS'] <= df.loc[iter_index_r, 'POS']:
                #     index_l = iter_index_l
                #     index_r = iter_index_r
                # else:
                #     index_l = iter_index_r
                #     index_r = iter_index_l

                # Check for overlaps
                if row_r['POS'] < row_l['END']:

                    # Check for record fully contained within another
                    if row_r['END'] <= row_l['END']:
                        # print('\t* Fully contained')

                        df.loc[index_r, 'INDEX'] = -1

                    else:

                        record_l, record_r, op_arr_l, op_arr_r = trim_alignment_record(
                            df.loc[index_l], df.loc[index_r],
                            'ref',
                            score_model=score_model,
                            op_arr_l=op_arr_l,
                            op_arr_r=op_arr_r
                        )

                        if record_l is not None and record_r is not None:

                            # Modify if new aligned size is at least min_trim_qry_len, remove if shorter
                            if record_l['QRY_END'] - record_l['QRY_POS'] >= min_trim_qry_len:
                                df.loc[index_l] = record_l
                                op_arr_dict[record_l['INDEX']] = op_arr_l
                            else:
                                df.loc[index_l, 'INDEX'] = -1

                            if (record_r['QRY_END'] - record_r['QRY_POS']) >= min_trim_qry_len:
                                df.loc[index_r] = record_r
                                op_arr_dict[record_r['INDEX']] = op_arr_r
                            else:
                                df.loc[index_r, 'INDEX'] = -1

                # Next r record
                iter_index_r += 1

            # Next l record
            iter_index_l += 1

        # Discard fully trimmed records
        df = df.loc[df['INDEX'] >= 0].copy()


    ###                      ###
    ### Post trim formatting ###
    ###                      ###

    # Clean and re-sort
    df = df.loc[(
            df['INDEX'] >= 0  # Removed records
        ) & (
            df['END'] - df['POS'] > 0  # Zero-length reference alignment (should not occur)
        ) & (
            df['QRY_END'] - df['QRY_POS'] > 0  # Zero-length query alignment (should not occur)
    )].copy()

    # df = df.loc[(df['END'] - df['POS']) > 0]  # Should never occur, but don't allow 0-length records
    # df = df.loc[(df['QRY_END'] - df['QRY_POS']) > 0]

    # Set query order (dropping records may have altered the order)
    df['QRY_ORDER'] = tables.get_qry_order(df)

    # Force-recompute features
    if len(op_arr_dict) > 0:
        df_alt = df.loc[df['INDEX'].isin(set(op_arr_dict.keys()))]
        df_same = df.loc[~ df.index.isin(set(df_alt.index))]

        op_arr_list = [
            op_arr_dict[index] for index in df_alt['INDEX']
        ]

        df = pd.concat([
            df_same,
            features.get_features(
                df=df_alt,
                feature_list=None,
                score_model=score_model,
                df_qry_fai=df_qry_fai,
                op_arr_list=op_arr_list,
                only_features=False,
                inplace=True,
                force_all=True
            )
        ], axis=0)

    # Re-sort in reference order
    df.sort_values(['#CHROM', 'POS', 'END', 'QRY_ID'], ascending=[True, True, False, True], inplace=True)

    # Check sanity
    df.apply(records.check_record, df_qry_fai=df_qry_fai, axis=1)

    # Return trimmed alignments
    return df


def get_op_arr_from_dict(record, op_arr_dict):
    """
    Pull an operation array from a dictionary or create a new one if it doesn't exist.

    :param record: Alignment record.
    :param op_arr_dict: Dictionary of operation arrays.

    :return: Operation array from dict or created from the record.
    """

    if record['INDEX'] not in op_arr_dict:
        return op.cigar_as_array(record['CIGAR'])

    return op_arr_dict[record['INDEX']]


def trim_alignment_record(
        record_l: pd.Series,
        record_r: pd.Series,
        match_coord: str,
        rev_l: bool=True,
        rev_r: bool=False,
        score_model: score.ScoreModel=None,
        op_arr_l: np.ndarray[int, int]=None,
        op_arr_r: np.ndarray[int, int]=None
) -> tuple[pd.Series, pd.Series, np.ndarray[int, int], np.ndarray[int, int]]:
    """
    Trim ends of overlapping alignments until ends no longer overlap. In repeat-mediated events, aligners may align the
    same parts of a query sequence to both reference copies (e.g. large DEL) or two parts of a query
    sequence to the same region (e.g. tandem duplication). This function trims back the alignments using the CIGAR
    string until the overlap is resolved using a simple greedy algorithm that maximizes the number of variants removed
    from the alignment during trimming (each variant is an insertion, deletion, or SNVs; currently, no bonus is given to
    removing larger insertions or deletions vs smaller ones).

    For example, a large repeat-mediated deletion will have two reference copies, but one copy in the query, and the
    single query copy is aligned to both by breaking the alignment record into two (one up to the deletion, and one
    following it). If the query coordinates were ignored, the alignment gap is smaller than the actual deletion event
    and one or both sides of the deletion are filled with false variants. In this example, the alignment is walked-
    out from both ends of the deletion until there is no duplication of aligned query (e.g. the alignment stops at
    one query base and picks up at the next query base). In this case, this function would be asked to resolve the
    query coordinates (match_coord = "query").

    A similar situation occurs for large tandem duplications, except there is one copy in the reference and two
    (or more) in the query. Aligners may align through the reference copy, break the alignment, and start a new
    alignment through the second copy in the query. In this case, this function would be asked to resolve reference
    coordinates (match_coord = "ref").

    :param record_l: Pandas Series alignment record (generated by align_get_read_bed). This record should be in the
        original alignment orientation regardless of whether `rev_l` is `True` or `False`.
    :param record_r: Pandas Series alignment record (generated by align_get_read_bed). This record should be in the
        original alignment orientation regardless of whether `rev_r` is `True` or `False`.
    :param match_coord: "qry" to trim query alignments, or "ref" to match reference alignments.
    :param rev_l: Trim `record_l` from the downstream end (alignment END) if `True`, otherwise, trim from the upstream
        end (alignment POS).
    :param rev_r: Trim `record_r` from the downstream end (alignment END) if `True`, otherwise, trim from the upstream
        end (alignment POS).
    :param score_model: Alignment model object or a configuration string describing a score model. If `None`, the
        default score model is used.
    :param op_arr_l: Alignment operation array for `record_l` or `None`.
    :param op_arr_r: Alignment operation array for `record_r` or `None`.

    :return: A tuple of four elements, the first two anre modified `record_l` and `record_r`, and the second two are
        trimmed alignment operation arrays for `record_l` and `record_r`.
    """

    if score_model is None:
        score_model = score.get_score_model()

    record_l = record_l.copy()
    record_r = record_r.copy()

    if op_arr_l is None:
        op_arr_l = op.cigar_as_array(record_l['CIGAR'])

    if op_arr_r is None:
        op_arr_r = op.cigar_as_array(record_r['CIGAR'])

    # Check arguments
    if match_coord == 'reference':
        match_coord = 'ref'

    if match_coord == 'query':
        match_coord = 'qry'

    if match_coord not in {'qry', 'ref'}:
        raise RuntimeError('Unknown match_coord parameter: {}: Expected "qry" or "ref"'.format(match_coord))

    # Determine if either record is filtered
    if 'FILTER' not in record_l:
        filter_l = False
    else:
        filter_l = not (record_l['FILTER'] == 'PASS' or pd.isnull(record_l['FILTER']))

    if 'FILTER' not in record_r:
        filter_r = False
    else:
        filter_r = not (record_r['FILTER'] == 'PASS' or pd.isnull(record_r['FILTER']))

    # Orient operations so regions to be trimmed are at the head of the list
    if rev_l:
        op_arr_l = op_arr_l[::-1]

    if rev_r:
        op_arr_r = op_arr_r[::-1]

    # Get number of bases to trim. Assumes records overlap.
    if match_coord == 'qry':

        diff_bp = np.min([record_l['QRY_END'], record_r['QRY_END']]) - np.max([record_l['QRY_POS'], record_r['QRY_POS']])

        if diff_bp < 0:
            raise RuntimeError(f'Cannot trim: records do not overlap in query space: {record_l["QRY_ID"]}:{record_l["QRY_POS"]}-{record_l["QRY_END"]} vs {record_r["QRY_ID"]}:{record_r["QRY_POS"]}-{record_r["QRY_END"]}')

    else:
        if record_l['POS'] > record_r['POS']:
            raise RuntimeError('Query sequences are incorrectly ordered in reference space: {} ({}:{}) vs {} ({}:{}), match_coord={}'.format(
                record_l['QRY_ID'], record_l['#CHROM'], record_l['POS'],
                record_r['QRY_ID'], record_r['#CHROM'], record_r['POS'],
                match_coord
            ))

        diff_bp = record_l['END'] - record_r['POS']

        if diff_bp <= 0:
            raise RuntimeError(f'Cannot trim: records do not overlap in reference space: {record_l["QRY_ID"]}:{record_l["QRY_POS"]}-{record_l["QRY_END"]} vs {record_r["QRY_ID"]}:{record_r["QRY_POS"]}-{record_r["QRY_END"]}')

    # Find the number of upstream (l) bases to trim to get to 0 (or query start)
    trace_l = trace_op_to_zero(op_arr_l, diff_bp, record_l, match_coord == 'qry', score_model)

    # Find the number of downstream (r) bases to trim to get to 0 (or query start)
    trace_r = trace_op_to_zero(op_arr_r, diff_bp, record_r, match_coord == 'qry', score_model)


    # For each upstream alignment cut-site, find the best matching downstream alignment cut-site. Not all cut-site
    # combinations need to be tested since trimmed bases and event count is non-decreasing as it moves away from the
    # best cut-site (residual overlapping bases 0 and maximum events consumed)

    if filter_l == filter_r:
        # Find optimal cut sites.
        # cut_idx_l and cut_idx_r are indices to trace_l and trace_r. These trace records point to the last alignment
        # operation to survive the cut, although they may be truncated (e.g. 100= to 90=).
        cut_idx_l, cut_idx_r = find_cut_sites(trace_l, trace_r, diff_bp, score_model)

    else:
        # One alignment record is filtered, but the other is not. Preferentially truncate the filtered alignment.
        if filter_l:
            cut_idx_l = len(trace_l) - 1
            cut_idx_r = 0
        else:
            cut_idx_l = 0
            cut_idx_r = len(trace_r) - 1

    # Check for no cut-sites. Should not occur at this stage.
    if cut_idx_l is None or cut_idx_r is None:
        raise RuntimeError('Program bug: Found no cut-sites: {} (INDEX={}) vs {} (INDEX={}), match_coord={}'.format(
            record_l['QRY_ID'], record_l['INDEX'],
            record_r['QRY_ID'], record_r['INDEX'],
            match_coord
        ))

    # Get cut records
    cut_l = trace_l[cut_idx_l]
    cut_r = trace_r[cut_idx_r]

    # Set mid-record cuts (Left-align cuts, mismatch first, preferentially trim filtered records)
    residual_bp = diff_bp - (cut_l[TC_DIFF_CUM] + cut_r[TC_DIFF_CUM])
    # trim_l = 0
    # trim_r = 0

    trim_dict = {'l': 0, 'r': 0}

    cut_dict = {'l': cut_l, 'r': cut_r}

    trim_order = {
        (False, False): [('l', op.X), ('r', op.X), ('l', op.EQ), ('r', op.EQ)],
        (True, True): [('l', op.X), ('r', op.X), ('l', op.EQ), ('r', op.EQ)],
        (True, False): [('l', op.X), ('l', op.EQ), ('r', op.X), ('r', op.EQ)],
        (False, True): [('r', op.X), ('r', op.EQ), ('l', op.X), ('l', op.EQ)],
    }

    for side, op_code in trim_order[(filter_l, filter_r)]:
        if residual_bp > 0 and cut_dict[side][TC_OP_CODE] == op_code:
            trim_bp = np.min([residual_bp, cut_dict[side][TC_OP_LEN] - 1])
            trim_dict[side] += trim_bp
            residual_bp -= trim_bp

    trim_l = trim_dict['l']
    trim_r = trim_dict['r']


    # if residual_bp > 0 and cut_r[TC_OP_CODE] == op.X:  # Right mismatch
    #     trim_r += np.min([residual_bp, cut_r[TC_OP_LEN] - 1])
    #     residual_bp -= trim_r
    #
    # if residual_bp > 0 and cut_l[TC_OP_CODE] == op.X:  # Left mismatch
    #     trim_l += np.min([residual_bp, cut_l[TC_OP_LEN] - 1])
    #     residual_bp -= trim_l
    #
    # if residual_bp > 0 and cut_l[TC_OP_CODE] == op.EQ:  # Left match
    #     trim_l += np.min([residual_bp, cut_l[TC_OP_LEN] - 1])
    #     residual_bp -= trim_l
    #
    # if residual_bp > 0 and cut_r[TC_OP_CODE] == op.EQ:  # Right match
    #     trim_r += np.min([residual_bp, cut_r[TC_OP_LEN] - 1])
    #     residual_bp -= trim_r

    # Get cut CIGAR String
    op_arr_l_mod = op_arr_l[cut_l[TC_INDEX]:]
    op_arr_r_mod = op_arr_r[cut_r[TC_INDEX]:]

    # Shorten last alignment record if set.
    op_arr_l_mod[0] = (op_arr_l_mod[0][0], op_arr_l_mod[0][1] - trim_l)
    op_arr_r_mod[0] = (op_arr_r_mod[0][0], op_arr_r_mod[0][1] - trim_r)

    # Modify alignment records
    record_l_mod = record_l.copy()
    record_r_mod = record_r.copy()

    cut_ref_l = cut_l[TC_REF_BP] + trim_l
    cut_qry_l = cut_l[TC_QRY_BP] + trim_l

    cut_ref_r = cut_r[TC_REF_BP] + trim_r
    cut_qry_r = cut_r[TC_QRY_BP] + trim_r

    if rev_l:
        record_l_mod['END'] -= cut_ref_l

        # Adjust positions in query space
        if record_l_mod['IS_REV']:
            record_l_mod['QRY_POS'] += cut_qry_l
        else:
            record_l_mod['QRY_END'] -= cut_qry_l

        # Track cut bases
        record_l_mod['TRIM_REF_R'] += cut_ref_l
        record_l_mod['TRIM_QRY_R'] += cut_qry_l

    else:
        record_l_mod['POS'] += cut_ref_l

        # Adjust positions in query space
        if record_l_mod['IS_REV']:
            record_l_mod['QRY_END'] -= cut_qry_l
        else:
            record_l_mod['QRY_POS'] += cut_qry_l

        # Track cut bases
        record_l_mod['TRIM_REF_L'] += cut_ref_l
        record_l_mod['TRIM_QRY_L'] += cut_qry_l

    if rev_r:
        record_r_mod['END'] -= cut_ref_r

        # Adjust positions in query space
        if record_r_mod['IS_REV']:
            record_r_mod['QRY_POS'] += cut_qry_r
        else:
            record_r_mod['QRY_END'] -= cut_qry_r

        # Track cut bases
        record_r_mod['TRIM_REF_R'] += cut_ref_r
        record_r_mod['TRIM_QRY_R'] += cut_qry_r

    else:
        record_r_mod['POS'] += cut_ref_r

        # Adjust positions in query space
        if record_r_mod['IS_REV']:
            record_r_mod['QRY_END'] -= cut_qry_r
        else:
            record_r_mod['QRY_POS'] += cut_qry_r

        # Track cut bases
        record_r_mod['TRIM_REF_L'] += cut_ref_r
        record_r_mod['TRIM_QRY_L'] += cut_qry_r

    # Add clipped bases to CIGAR
    if cut_l[TC_CLIPH_BP] > 0:
        cigar_l_pre = [(op.H, cut_l[TC_CLIPH_BP])]
    else:
        cigar_l_pre = []

    if cut_r[TC_CLIPH_BP] > 0:
        cigar_r_pre = [(op.H, cut_r[TC_CLIPH_BP])]
    else:
        cigar_r_pre = []

    clip_s_l = cut_l[TC_CLIPS_BP] + cut_l[TC_QRY_BP] + trim_l
    clip_s_r = cut_r[TC_CLIPS_BP] + cut_r[TC_QRY_BP] + trim_r

    if clip_s_l > 0:
        cigar_l_pre.append((op.S, clip_s_l))

    if clip_s_r > 0:
        cigar_r_pre.append((op.S, clip_s_r))

    # Append remaining CIGAR
    if len(cigar_l_pre) > 0:
        op_arr_l_mod = np.append(cigar_l_pre, op_arr_l_mod, axis=0)

    if len(cigar_r_pre) > 0:
        op_arr_r_mod = np.append(cigar_r_pre, op_arr_r_mod, axis=0)

    # Finish CIGAR and update score
    if rev_l:
        op_arr_l_mod = op_arr_l_mod[::-1]

    if rev_r:
        op_arr_r_mod = op_arr_r_mod[::-1]

    record_l_mod['CIGAR'] = op.to_cigar_string(op_arr_l_mod)
    record_l_mod['SCORE'] = score_model.score_operations(op_arr_l_mod)

    record_r_mod['CIGAR'] = op.to_cigar_string(op_arr_r_mod)
    record_r_mod['SCORE'] = score_model.score_operations(op_arr_r_mod)

    # Return trimmed records
    return record_l_mod, record_r_mod, op_arr_l_mod, op_arr_r_mod


def find_cut_sites(
        trace_l: list[tuple],
        trace_r: list[tuple],
        diff_bp: int,
        score_model: score.ScoreModel
):
    """
    Find best cut-sites for left and right alignments to consume `diff_bp` bases.

    Optimize by:
    1) `diff_bp` or more bases removed (avoid over-trimming)
    2) Minimize lost score (prefer dropping negative scoring events)
    3) Tie-break by:
      a) Total removed bases closest to `diff_bp`.
      b) Left-align break (trace_l is preferentially trimmed when there is a tie).

    :param trace_l: List of tuples for the left alignment generated by `trace_op_to_zero()`.
    :param trace_r: List of tuples for the right alignment generated by `trace_op_to_zero()`.
    :param diff_bp: Target removing this many bases. Could be reference or query depending on how the
        traces were constructed.
    :param score_model: Alignment score model.

    :return: Tuple of (cut_idx_l, cut_idx_r). cut_idx_l and cut_idx_r are the left query and right query operation list
        index (argument to trace_op_to_zero()), index element of `trace_l` and `trace_r`) where the alignment cuts
        should occur.
    """

    # Right-index traversal
    tc_idx_r = 0         # Current right-index in trace record list (tc)

    len_r = len(trace_r)  # End of r-indexes

    # Optimal cut-site for this pair of alignments
    cut_idx_l = None  # Record where cut occurs in left trace
    cut_idx_r = None  # Record where cut occurs in right trace

    min_score = np.inf      # Minimum cumulative score that may be cut
    max_diff_optimal = None  # Optimal difference in the number of bases cut over diff_bp. closest to 0 means cut-site
                             # can be placed exactly and does not force over-cutting to remove overlap)

    # Traverse l cut-sites
    for tc_idx_l in range(len(trace_l) - 1, -1, -1):

        # Get min and max base differences achievable by cutting at the end or beginning of this l-record.
        min_bp_l = trace_l[tc_idx_l][TC_DIFF_CUM]
        max_bp_l = trace_l[tc_idx_l][TC_DIFF_CUM] + trace_l[tc_idx_l][TC_DIFF] - 1  # Cut all but one left base

        # Traverse r cut-sites until max-left + max-right base difference diff_bp or greater.
        while (
                tc_idx_r + 1 < len_r and
                max_bp_l + trace_r[tc_idx_r][TC_DIFF_CUM] + trace_r[tc_idx_r][TC_DIFF] - 1 < diff_bp  # Cut all but one right base
        ):
            tc_idx_r += 1

        # Traverse all cases where max-cutting the left event crosses 0 residual bases (or the single case resulting in
        # over-cutting). After this loop, the range of acceptable right indices spans tc_idx_r_start to tc_idx_r (exclusive on right side).
        tc_idx_r_start = tc_idx_r

        while (
                tc_idx_r < len_r and (
                    min_bp_l + trace_r[tc_idx_r][TC_DIFF_CUM] <= diff_bp or  # Acceptable cut site not found
                    tc_idx_r == tc_idx_r_start  # Find at least one cut-site on the right side, even if it over-cuts.
                )
        ):

            # Collect cut-site stats
            #min_bp = min_bp_l + trace_r[tc_idx_r][TC_DIFF_CUM]
            max_bp = max_bp_l + trace_r[tc_idx_r][TC_DIFF_CUM] + trace_r[tc_idx_r][TC_DIFF] - 1

            diff_min = diff_bp - max_bp

            # Count number of events if the minimal cut at these sites are made.
            score_cum = trace_l[tc_idx_l][TC_SCORE_CUM] + trace_r[tc_idx_r][TC_SCORE_CUM]

            if diff_min <= 0:

                residual_x = np.min([
                    np.abs(diff_min),
                    (
                        trace_l[tc_idx_l][TC_OP_LEN] - 1 if trace_l[tc_idx_l][TC_OP_CODE] == op.X else 0
                    ) + (
                        trace_r[tc_idx_r][TC_OP_LEN] - 1 if trace_r[tc_idx_r][TC_OP_CODE] == op.X else 0
                    )
                ])

                diff_min += residual_x

                residual_eq = np.min([
                    np.abs(diff_min),
                    (
                        trace_l[tc_idx_l][TC_OP_LEN] - 1 if trace_l[tc_idx_l][TC_OP_CODE] == op.EQ else 0
                    ) + (
                        trace_r[tc_idx_r][TC_OP_LEN] - 1 if trace_r[tc_idx_r][TC_OP_CODE] == op.EQ else 0
                    )
                ])

                score_cum += score_model.score(op.X, residual_x) + score_model.score(op.EQ, residual_eq)

                diff_optimal = 0  # diff_bp is exactly achievable
            else:
                # Must over-cut to use these sites.
                diff_optimal = diff_min

            # # Save max
            # if (
            #     score_cum < min_score_part or (  # Better alignment score, or
            #         score_cum == min_score_part and (  # Same event count, and
            #             max_diff_optimal_part is None or diff_optimal < max_diff_optimal_part  # Optimal difference is closer to 0 (less over-cut)
            #         )
            #     )
            # ):
            #     cut_idx_part_l = tc_idx_l
            #     cut_idx_part_r = tc_idx_r
            #     min_score_part = score_cum
            #     max_diff_optimal_part = diff_optimal
            #
            # tc_idx_r += 1

            # Save max
            if (
                score_cum < min_score or (  # Better event count, or
                    score_cum == min_score and (  # Same event count, and
                        max_diff_optimal is None or diff_optimal < max_diff_optimal  # Optimal difference is closer to 0 (less over-cut)
                    )
                )
            ):
                cut_idx_l = tc_idx_l
                cut_idx_r = tc_idx_r
                min_score = score_cum
                max_diff_optimal = diff_optimal

            tc_idx_r += 1

        # Reset right index
        tc_idx_r = tc_idx_r_start

    return cut_idx_l, cut_idx_r


def trace_op_to_zero(
        op_arr: np.ndarray[int, int],
        diff_bp: int,
        aln_record: pd.Series,
        diff_query: bool,
        score_model: score.ScoreModel
) -> list[tuple]:
    """
    Trace align operations back until diff_bp query bases are discarded from the alignment. Operations must only
    contain operators "IDSH=X" (no "M"). The array returned is only alignment match ("=" or "X" records) for the
    optimal-cut algorithm (can only cut at aligned bases).

    Returns a list of tuples for each operation traversed:
        * TC_INDEX = 0: Index in cigar_list.
        * TC_OP_CODE = 1: operation code (character, e.g. "I", "=").
        * TC_OP_LEN = 2: Operation length.
        * TC_DIFF_CUM = 3: Cumulative base difference up this event, but not including it.
        * TC_DIFF = 4: Base difference for this event. Will be op_len depending on the operation code.
        * TC_EVENT_CUM = 5: Cumulative event difference (number of insertions, deletions, and SNVs) up to this event,
            but not including it.
        * TC_EVENT = 6: Event differences for this event. "1" for insertions or deletions, OP_LEN for mismatches
            "X", SNV).
        * TC_REF_BP = 7: Cumulative number of reference bases consumed up to this event, but not including it.
        * TC_QRY_BP = 8: Cumulative number of query bases consumed up to this event, but not including it.
        * TC_CLIPS_BP = 9: Cumulative number of soft-clipped bases up to AND INCLUDING this event. Alignments are not
            cut on clipped records, so cumulative and including does not affect the algorithm.
        * TC_CLIPH_BP = 10: Cumulative number of hard-clipped bases up to AND INCLUDING this event.
        * TC_SCORE_CUM = 11: Cumulative alignment score up to this event, but not including it.

    :param op_arr: Array of alignment operations (col 1: cigar_len, col 2: cigar_op).
    :param diff_bp: Number of query bases to trace back. Final record will traverse past this value.
    :param aln_record: Alignment record for error reporting.
    :param diff_query: Compute base differences for query sequence if `True`. If `False`, compute for reference.
    :param score_model: Alignment score model.

    :return: A list of tuples tracing the effects of truncating an alignment at a given CIGAR operation.
    """

    index = 0
    index_end = op_arr.shape[0]

    op_count = 0

    diff_cumulative = 0
    event_cumulative = 0
    score_cumulative = 0

    ref_bp_sum = 0
    qry_bp_sum = 0
    clip_s_sum = 0
    clip_h_sum = 0

    trace_list = list()

    last_no_match = False  # Continue until the last element is a match

    while index < index_end and (diff_cumulative <= diff_bp or last_no_match or len(trace_list) == 0):
        op_count += 1
        op_code, op_len = op_arr[index]

        last_no_match = True

        if op_code == op.EQ:
            event_count = 0

            ref_bp = op_len
            qry_bp = op_len

            last_no_match = False

        elif op_code == op.X:
            event_count = op_len

            ref_bp = op_len
            qry_bp = op_len

        elif op_code == op.I:
            event_count = 1

            ref_bp = 0
            qry_bp = op_len

        elif op_code == op.D:
            event_count = 1

            ref_bp = op_len
            qry_bp = 0

        elif op_code == op.S:
            event_count = 0

            ref_bp = 0
            qry_bp = 0

            clip_s_sum += op_len

        elif op_code == op.H:
            event_count = 0

            ref_bp = 0
            qry_bp = 0

            clip_h_sum += op_len

        else:
            raise RuntimeError((
                'Illegal alignment operation while trimming alignment: {} '
                '(start={}:{}): Operation #{}: Expected op in "IDSH=X"'
            ).format(op_code, aln_record['#CHROM'], aln_record['POS'], index))

        # Get number of bases affected by this event
        if diff_query:
            diff_change = qry_bp
        else:
            diff_change = ref_bp

        # Add to trace list
        if op_code in op.EQX_SET:
            trace_list.append(
                (
                    index,
                    op_code, op_len,
                    diff_cumulative, diff_change,
                    event_cumulative, event_count,
                    ref_bp_sum, qry_bp_sum,
                    clip_s_sum, clip_h_sum,
                    score_cumulative
                )
            )

        # Increment cumulative counts
        diff_cumulative += diff_change
        event_cumulative += event_count
        score_cumulative += score_model.score(op_code, op_len)

        ref_bp_sum += ref_bp
        qry_bp_sum += qry_bp

        index += 1

    return trace_list


#
# Truncate overlapping records
#

def truncate_alignment_record(record, overlap_bp, trunc_side, score_model=None, df_qry_fai=None):
    """
    Truncate overlapping alignments in query space modifying the alignment record in-place. Similar to alignment
    trimming except only one side is trimmed.

    The alignment record is trimmed from either side (left "l" for QRY_POS up to `trunc_pos`, or right "r" for QRY_END back
    to `pos`). A modified alignment record is returned.

    :param record: Alignment record to truncate.
    :param overlap_bp: Number of query bases to trim.
    :param trunc_side: Trim from this side of the record. Valid values are "l" for left side and "r" for right side.
    :param score_model: Alignment model object (`pavlib.align.score.ScoreModel`) or a configuration string to generate
        a score model object. If `None`, the default score model is used. An alignment score is computed by summing
        the score of each CIGAR operation against this model (match, mismatch, and gap) to update the "SCORE" column.
    :param df_qry_fai: DataFrame of query FASTA index (FAI) file or None. Used if a feature in
        pavlib.align.util.ALIGN_FEATURE_COLUMNS need the query length.

    :return: A modified alignment record or None if the alignment was completely truncated.
    """

    score_model = score.get_score_model(score_model)

    # Check arguments
    if trunc_side not in {'l', 'r'}:
        raise ValueError(f'Invalid argument for "trunc_side" (expected "l" or "r"): {trunc_side}')

    if overlap_bp <= 0:
        raise RuntimeError(f'Overlap bp must be positive: {overlap_bp}')

    # Get cigar operations (list of (op_len, op_code) tuples)
    op_arr = op.cigar_as_array(record)

    # Orient CIGAR operations so regions to be trimmed are at the head of the list
    is_rev = record['IS_REV'] ^ (trunc_side == 'l')

    if is_rev:
        op_arr = op_arr[::-1]

    # Init cut site search
    op_loc = op_arr.shape[0] - 1

    cut_bp_qry = 0
    cut_bp_ref = 0

    clip_bp = 0

    # Eliminate clipping
    while op_loc > 0 and op_arr[op_loc, 1] in op.CLIP_SET:
        clip_bp += op_arr[op_loc, 0]
        op_loc -= 1

    if op_loc < 0:
        raise RuntimeError(f'No records to trim')

    # Find cut site
    while op_loc > 0 and op_arr[op_loc, 1] not in op.CLIP_SET:
        op_code, op_len = op_arr[op_loc]

        if op_code in op.EQX_SET:
            if cut_bp_qry + op_len > overlap_bp:
                break

            cut_bp_qry += op_len
            cut_bp_ref += op_len

        elif op_code == op.I:
            cut_bp_qry += op_len

        elif op_code == op.D:
            cut_bp_ref += op_len

        else:
            raise RuntimeError((
                'Illegal operation in query alignment while truncating alignment: {} '
                '(alignment record={}): Expected CIGAR op in "IDSH=X"'
            ).format(op_code, record['INDEX']))

        op_loc -= 1

    # Trim mid-record
    if cut_bp_qry < overlap_bp:

        residual_bp = overlap_bp - cut_bp_qry

        # Check CIGAR position
        if op_loc < 0:
            raise RuntimeError(f'Ran out of CIGAR operations: {residual_bp} bp left with no more CIGAR operations to trim')

        if op_arr[op_loc, 0] not in op.EQX_SET:
            raise RuntimeError(f'Ran out of CIGAR operations: {residual_bp} bp with only clipping CIGAR operations remaining')

        if op_arr[op_loc, 0] not in op.EQX_SET:
            raise RuntimeError(f'Program bug: Mid-operation cut site is not in an "=" or "X" record: (op_code={op_arr[op_loc, 0]})')

        if op_arr[op_loc, 1] <= residual_bp:
            raise RuntimeError(f'Program bug: Mid-operation cut is not less than than the "=" or "X" record size: (residual={residual_bp}, op_code={op_arr[op_loc, 0]})')

        # Truncate CIGAR records before this one
        op_arr = op_arr[0:op_loc + 1]

        # Modify record, shuffle trimmed bases to clipping
        op_arr[-1, 1] -= residual_bp

        cut_bp_qry += residual_bp
        cut_bp_ref += residual_bp

    else:
        if op_loc < 0:
            return None

    op_arr = op_arr[0:op_loc + 1]

    # Truncate all records?
    if op_arr[-1, 0] in op.CLIP_SET:
        for cigar_loc2 in range(op_loc):
            if op_arr[cigar_loc2, 0] not in op.CLIP_SET:
                raise RuntimeError(f'Found clipping mid-CIGAR')

        return None

    # Add clipping
    if cut_bp_qry == 0:
        raise RuntimeError('Program Bug: Overlap detected, but no truncation was performed')

    op_arr = np.append(op_arr, [op.H, clip_bp + cut_bp_qry], axis=0)

    # Modify record
    if is_rev:
        op_arr = op_arr[::-1]

    record['CIGAR'] = op.to_cigar_string(op_arr)

    if record['IS_REV'] ^ (trunc_side == 'r'):
        record['END'] -= cut_bp_ref
        record['TRIM_REF_R'] += cut_bp_ref
        record['TRIM_QRY_R'] += cut_bp_qry
    else:
        record['POS'] += cut_bp_ref
        record['TRIM_REF_L'] += cut_bp_ref
        record['TRIM_QRY_L'] += cut_bp_qry

    if trunc_side == 'r':
        record['QRY_END'] -= cut_bp_qry
    else:
        record['QRY_POS'] += cut_bp_qry

    record = features.get_features(
        df=record,
        feature_list=features.ALIGN_FEATURE_COLUMNS,
        score_model=score_model,
        existing_score_model=False,
        score_prop_conf=features.ALIGN_FEATURE_SCORE_PROP_CONF,
        df_qry_fai=df_qry_fai,
        only_features=False
    )

    return record
