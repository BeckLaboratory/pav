# Alignment trimming functions

import collections
import intervaltree
import numpy as np
import pandas as pd

from typing import Any

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
TC_REF_BP = 5
TC_QRY_BP = 6
TC_CLIP_BP = 7
TC_SCORE_CUM = 8

TRIM_DESC = {
    'none': 'No trimming',
    'qry': 'Query-only trimming',
    'qryref': 'Query-Reference trimming'
}

COORD_ARR_COLS = ['POS', 'END', 'QRY_POS', 'QRY_END', 'TRIM_REF_L', 'TRIM_REF_R', 'TRIM_QRY_L', 'TRIM_QRY_R']

COORD_POS = 0
COORD_END = 1
COORD_QRY_POS = 2
COORD_QRY_END = 3
COORD_TRIM_REF_L = 4
COORD_TRIM_REF_R = 5
COORD_TRIM_QRY_L = 6
COORD_TRIM_QRY_R = 7

SEQNAME_ARR_COLS = ['#CHROM', 'QRY_ID']

SEQNAME_CHROM = 0
SEQNAME_QRY_ID = 1


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
    op_arr_list = [None] * df.shape[0]

    # Array of alignment records that have not been eliminated
    active_records = np.ones(df.shape[0], dtype=bool)

    # Remove short alignments
    active_records[df['QRY_END'] - df['QRY_POS'] < min_trim_qry_len] = False

    # Save fields as array for fast access (__getitem__ in Pandas as extremely slow)
    coord_arr = df[COORD_ARR_COLS].values
    is_rev_arr = df['IS_REV'].values
    seqname_arr = df[SEQNAME_ARR_COLS].values.astype(str)
    score_arr = df['SCORE'].values

    if 'FILTER' in df.columns:
        is_filter_fail_arr = ~ ((df['FILTER'] == 'PASS') | df['FILTER'].isna()).values
    else:
        is_filter_fail_arr = np.zeros(df.shape[0], dtype=bool)

    def get_op_arr(index:int) -> np.ndarray:
        if op_arr_list[index] is None:
            op_arr_list[index] = op.cigar_as_array(df.iloc[index]['CIGAR'])

        return op_arr_list[index]

    #
    # Trim in query coordinates
    #

    if do_trim_qry:
        if pav_params.debug:
            print(f'Trimming in query coordinates', flush=True)

        # Sort by alignment lengths in query coordinates
        record_order_qry_id = df.reset_index(drop=True).sort_values(['QRY_ID', 'SCORE'], ascending=(True, False))['QRY_ID']

        for qry_id in record_order_qry_id.unique():
            if pav_params.debug:
                print(f'Trimming query: {qry_id}')

            qry_index = record_order_qry_id[record_order_qry_id == qry_id].index.values\

            depth_filter_set = set(
                filter_depth(
                    record_index=qry_index,
                    coord_arr=coord_arr,
                    is_qry=True,
                    max_depth=pav_params.align_trim_max_depth,
                    max_overlap=pav_params.align_trim_max_depth_prop
                )
            )

            # Build a tree of intersecting alignments
            itree = intervaltree.IntervalTree()

            iter_index_r = 0

            while iter_index_r < qry_index.shape[0]:

                base_index_r = qry_index[iter_index_r]

                if base_index_r not in depth_filter_set:
                    active_records[base_index_r] = False
                    iter_index_r += 1
                    continue

                if not active_records[base_index_r]:
                    iter_index_r += 1
                    continue

                for index_l in qry_index[
                    sorted({
                        match.data for match in
                            itree[
                                coord_arr[base_index_r, COORD_QRY_POS]:
                                coord_arr[base_index_r, COORD_QRY_END]
                            ]
                    })
                ]:

                    index_r = base_index_r  # May have been swapped with index_l in the previous iteration

                    if not active_records[index_l] or not active_records[index_r]:
                        continue

                    # Switch indices to sequence order
                    if (
                        coord_arr[index_l, COORD_QRY_POS] > coord_arr[index_r, COORD_QRY_POS]  # R comes before L
                    ) or (
                        coord_arr[index_l, COORD_QRY_POS] == coord_arr[index_r, COORD_QRY_POS] and  # R and L start at the same position...
                        coord_arr[index_l, COORD_QRY_END] < coord_arr[index_r, COORD_QRY_END]       # ...but L is shorter
                    ):
                        index_l, index_r = index_r, index_l

                    # Skip if there is no overlap
                    if coord_arr[index_r, COORD_QRY_POS] >= coord_arr[index_l, COORD_QRY_END]:
                        continue

                    # Check if record is contained within another
                    if coord_arr[index_r, COORD_QRY_END] <= coord_arr[index_l, COORD_QRY_END]:

                        if is_filter_fail_arr[index_l] and not is_filter_fail_arr[index_r]:
                            active_records[index_l] = False
                        else:
                            active_records[index_r] = False

                        continue

                    # Determine trim orientation (right side of index_l is to be trimmed, must be reversed so
                    # trimmed alignment records are at the beginning; left side of index_r is to be trimmed, which is
                    # already at the start of the alignment operation list).
                    rev_l = not is_rev_arr[index_l]  # Trim right end of index_l
                    rev_r = is_rev_arr[index_r]      # Trim left end of index_r

                    # Get operation arrays
                    op_arr_l = get_op_arr(index_l)
                    op_arr_r = get_op_arr(index_r)

                    # Determine which side to preferentially trim in case of ties
                    if (
                        rev_l != rev_r  # Same as row_l['IS_REV'] == row_r['IS_REV']
                    ) and (
                        seqname_arr[index_l, SEQNAME_CHROM] == seqname_arr[index_r, SEQNAME_CHROM]  # Same chromosome
                    ):
                        # Same chromosome and orientation, preferentially trim left-most record to maintain left-aligning
                        # breakpoints.

                        prefer_l = (
                            coord_arr[index_l, COORD_QRY_POS] if is_rev_arr[index_l] else coord_arr[index_l, COORD_QRY_END]  # Reference position on left record being trimmed
                        ) <= (
                            coord_arr[index_r, COORD_QRY_END] if is_rev_arr[index_r] else coord_arr[index_r, COORD_QRY_POS]  # Reference position on right record being trimmed
                        )

                    else:
                        prefer_l = score_arr[index_l] <= score_arr[index_r]

                    # Do trimming
                    try:
                        if prefer_l:
                            op_arr_l_mod, op_arr_r_mod = trim_alignment_record(
                                index_l_trim=index_l,
                                index_r_trim=index_r,
                                rev_l_trim=rev_l,
                                rev_r_trim=rev_r,
                                coord_arr=coord_arr,
                                is_filter_fail_arr=is_filter_fail_arr,
                                match_qry=True,
                                is_rev_arr=is_rev_arr,
                                score_model=score_model,
                                op_arr_l=op_arr_l,
                                op_arr_r=op_arr_r
                            )
                        else:
                            op_arr_r_mod, op_arr_l_mod = trim_alignment_record(
                                index_l_trim=index_r,
                                index_r_trim=index_l,
                                rev_l_trim=rev_r,
                                rev_r_trim=rev_l,
                                coord_arr=coord_arr,
                                is_filter_fail_arr=is_filter_fail_arr,
                                match_qry=True,
                                is_rev_arr=is_rev_arr,
                                score_model=score_model,
                                op_arr_l=op_arr_r,
                                op_arr_r=op_arr_l
                            )
                    except Exception as e:
                        raise RuntimeError(
                            f'Error trimming overlapping alignments in query coordinates: '
                            f'INDEX {df.iloc[index_l]["INDEX"]} ({seqname_arr[index_l, SEQNAME_CHROM]}:{coord_arr[index_l, COORD_POS]:,}-{coord_arr[index_l, COORD_END]:,}, {seqname_arr[index_l, SEQNAME_QRY_ID]}:{coord_arr[index_l, COORD_QRY_POS]:,}-{coord_arr[index_l, COORD_QRY_END]:,}), '
                            f'INDEX {df.iloc[index_r]["INDEX"]} ({seqname_arr[index_r, SEQNAME_CHROM]}:{coord_arr[index_r, COORD_POS]:,}-{coord_arr[index_r, COORD_END]:,}, {seqname_arr[index_r, SEQNAME_QRY_ID]}:{coord_arr[index_r, COORD_QRY_POS]:,}-{coord_arr[index_r, COORD_QRY_END]:,}): '
                            f'Table indices {index_l}, {index_r} (QRY_ID={qry_id}): '
                            f'{e}'
                        )

                    # Check trimming
                    if coord_arr[index_r, COORD_QRY_POS] < coord_arr[index_l, COORD_QRY_END]:
                        raise RuntimeError(
                            f'Found overlapping query bases after trimming in query coordinates: '
                            f'INDEX {df.iloc[index_l]["INDEX"]} (QRY_END={coord_arr[index_l, COORD_QRY_END]:,}), '
                            f'INDEX {df.iloc[index_r]["INDEX"]} (QRY_POS={coord_arr[index_r, COORD_QRY_POS]:,}): '
                            f'Table indices {index_l}, {index_r} (QRY_ID={qry_id})'
                        )

                    # Save operation arrays
                    op_arr_list[index_l] = op_arr_l_mod
                    op_arr_list[index_r] = op_arr_r_mod

                    # Modify if new aligned size is at least min_trim_qry_len, remove if shorter
                    if coord_arr[index_l, COORD_QRY_END] - coord_arr[index_l, COORD_QRY_POS] < min_trim_qry_len:
                        active_records[index_l] = False

                    if coord_arr[index_r, COORD_QRY_END] - coord_arr[index_r, COORD_QRY_POS] < min_trim_qry_len:
                        active_records[index_r] = False

                # Save index
                itree[
                    coord_arr[base_index_r, COORD_QRY_POS]:
                    coord_arr[base_index_r, COORD_QRY_END]
                ] = iter_index_r

                iter_index_r += 1


    #
    # Trim in reference coordinates
    #

    if do_trim_ref:
        if pav_params.debug:
            print(f'Trimming reference coordinates', flush=True)

        # Sort by alignment lengths in query space
        record_order_chrom = df.reset_index(drop=True).sort_values(['#CHROM', 'SCORE'], ascending=(True, False))['#CHROM']

        for chrom in record_order_chrom.unique():
            if pav_params.debug:
                print(f'Trimming chrom: {chrom}', flush=True)

            chrom_index = record_order_chrom[record_order_chrom == chrom].index.values

            depth_filter_set = set(  # Set of indices passing the depth filter
                filter_depth(
                    record_index=chrom_index,
                    coord_arr=coord_arr,
                    is_qry=False,
                    max_depth=pav_params.align_trim_max_depth,
                    max_overlap=pav_params.align_trim_max_depth_prop
                )
            )

            # Build a tree of intersecting alignments
            itree = intervaltree.IntervalTree()

            iter_index_r = 0

            while iter_index_r < chrom_index.shape[0]:

                base_index_r = chrom_index[iter_index_r]

                if base_index_r not in depth_filter_set:
                    active_records[base_index_r] = False
                    iter_index_r += 1
                    continue

                if not active_records[base_index_r]:
                    iter_index_r += 1
                    continue

                for index_l in chrom_index[
                    sorted({
                        match.data for match in
                            itree[
                                coord_arr[base_index_r, COORD_POS]:
                                coord_arr[base_index_r, COORD_END]
                            ]
                    })
                ]:

                    index_r = base_index_r  # May have been swapped with index_l in the previous iteration

                    if not active_records[index_l] or not active_records[index_r]:
                        continue

                    # Skip if match_qry and query names differ
                    if match_qry and seqname_arr[index_l, SEQNAME_QRY_ID] != seqname_arr[index_r, SEQNAME_QRY_ID]:
                        continue

                    # Switch indices to sequence order
                    if (
                        coord_arr[index_l, COORD_POS] > coord_arr[index_r, COORD_POS]  # R comes before L
                    ) or (
                        coord_arr[index_l, COORD_POS] == coord_arr[index_r, COORD_POS] and  # R and L start at the same position...
                        coord_arr[index_l, COORD_END] < coord_arr[index_r, COORD_END]       # ...but L is shorter
                    ):
                        index_l, index_r = index_r, index_l

                    # Skip if no overlap
                    if coord_arr[index_r, COORD_POS] >= coord_arr[index_l, COORD_END]:
                        continue

                    # Check if record is contained within another
                    if coord_arr[index_r, COORD_END] <= coord_arr[index_l, COORD_END]:

                        if is_filter_fail_arr[index_l] and not is_filter_fail_arr[index_r]:
                            active_records[index_l] = False
                        else:
                            active_records[index_r] = False

                        continue

                    # Get operation arrays
                    op_arr_l = get_op_arr(index_l)
                    op_arr_r = get_op_arr(index_r)

                    # Trim records
                    try:
                        op_arr_l_mod, op_arr_r_mod = trim_alignment_record(
                            index_l_trim=index_l,
                            index_r_trim=index_r,
                            rev_l_trim=True,
                            rev_r_trim=False,
                            coord_arr=coord_arr,
                            is_filter_fail_arr=is_filter_fail_arr,
                            match_qry=False,
                            is_rev_arr=is_rev_arr,
                            score_model=score_model,
                            op_arr_l=op_arr_l,
                            op_arr_r=op_arr_r
                        )

                    except Exception as e:
                        raise RuntimeError(
                            f'Error trimming overlapping alignments in reference coordinates: '
                            f'INDEX {df.iloc[index_l]["INDEX"]} ({seqname_arr[index_l, SEQNAME_CHROM]}:{coord_arr[index_l, COORD_POS]:,}-{coord_arr[index_l, COORD_END]:,}, {seqname_arr[index_l, SEQNAME_QRY_ID]}:{coord_arr[index_l, COORD_QRY_POS]:,}-{coord_arr[index_l, COORD_QRY_END]:,}), '
                            f'INDEX {df.iloc[index_r]["INDEX"]} ({seqname_arr[index_r, SEQNAME_CHROM]}:{coord_arr[index_r, COORD_POS]:,}-{coord_arr[index_r, COORD_END]:,}, {seqname_arr[index_r, SEQNAME_QRY_ID]}:{coord_arr[index_r, COORD_QRY_POS]:,}-{coord_arr[index_r, COORD_QRY_END]:,}): '
                            f'Table indices {index_l}, {index_r} (QRY_ID={qry_id}): '
                            f'{e}'
                        )

                    # Check trimming
                    if coord_arr[index_r, COORD_POS] < coord_arr[index_l, COORD_POS]:
                        raise RuntimeError(
                            f'Found overlapping query bases after trimming in reference coordinates: '
                            f'INDEX {df.iloc[index_l]["INDEX"]} (END={coord_arr[index_l, COORD_END]:,}), '
                            f'INDEX {df.iloc[index_r]["INDEX"]} (POS={coord_arr[index_r, COORD_POS]:,}): '
                            f'Table indices {index_l}, {index_r} (CHROM={chrom})'
                        )

                    # Save operation arrays
                    op_arr_list[index_l] = op_arr_l_mod
                    op_arr_list[index_r] = op_arr_r_mod

                    # Modify if new aligned size is at least min_trim_qry_len, remove if shorter
                    if coord_arr[index_l, COORD_QRY_END] - coord_arr[index_l, COORD_QRY_POS] < min_trim_qry_len:
                        active_records[index_l] = False

                    if coord_arr[index_r, COORD_QRY_END] - coord_arr[index_r, COORD_QRY_POS] < min_trim_qry_len:
                        active_records[index_r] = False

                # Save index
                itree[
                    coord_arr[base_index_r, COORD_POS]:
                    coord_arr[base_index_r, COORD_END]
                ] = iter_index_r

                iter_index_r += 1


    #
    # Post trim formatting
    #

    if pav_params.debug:
        print(f'Post trim: Updating records', flush=True)

    # Set CIGAR string
    df['CIGAR'] = [
        op.to_cigar_string(op_arr_list[i]) if (
            op_arr_list[i] is not None and active_records[i]
        ) else df.iloc[i]['CIGAR']
            for i in range(df.shape[0])
    ]

    # Update table with modified coordinates
    df['POS'] = coord_arr[:, COORD_POS]
    df['END'] = coord_arr[:, COORD_END]
    df['QRY_POS'] = coord_arr[:, COORD_QRY_POS]
    df['QRY_END'] = coord_arr[:, COORD_QRY_END]
    df['TRIM_REF_L'] = coord_arr[:, COORD_TRIM_REF_L]
    df['TRIM_REF_R'] = coord_arr[:, COORD_TRIM_REF_R]
    df['TRIM_QRY_L'] = coord_arr[:, COORD_TRIM_QRY_L]
    df['TRIM_QRY_R'] = coord_arr[:, COORD_TRIM_QRY_R]

    # Split modified records
    if pav_params.debug:
        print(f'Post trim: Updating table and recalculating features', flush=True)

    mod_records = np.array([op_arr is not None for op_arr in op_arr_list])

    df_mod = df[mod_records & active_records]
    df_nmod = df[~mod_records & active_records]

    df = pd.concat([
        df_nmod,
        features.get_features(
            df=df_mod,
            feature_list=None,
            score_model=score_model,
            df_qry_fai=df_qry_fai,
            op_arr_list=[op_arr_list[i] for i in range(df.shape[0]) if mod_records[i] and active_records[i]],
            only_features=False,
            inplace=True,
            force_all=True
        )
    ], axis=0)

    # Re-sort in reference order
    df.sort_values(['#CHROM', 'POS', 'END', 'QRY_ID'], ascending=[True, True, False, True], inplace=True)
    df['QRY_ORDER'] = tables.get_qry_order(df)

    # Check sanity
    if pav_params.debug:
        print(f'Post trim: Verifying records', flush=True)

    df.apply(records.check_record, df_qry_fai=df_qry_fai, axis=1)

    # Return trimmed alignments
    if pav_params.debug:
        print(f'Trimming complete', flush=True)

    return df


def trim_alignment_record(
        index_l_trim: int,
        index_r_trim: int,
        rev_l_trim: bool,
        rev_r_trim: bool,
        match_qry: bool,
        op_arr_l: np.ndarray,
        op_arr_r: np.ndarray,
        coord_arr: np.ndarray,
        is_filter_fail_arr: np.ndarray,
        is_rev_arr: np.ndarray,
        score_model: score.ScoreModel
) -> tuple[np.ndarray, np.ndarray]:
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

    :param index_l_trim: Index of `record_l` to trim.
    :param index_r_trim: Index of `record_r` to trim.
    :param rev_l_trim: True if `record_l` is reversed for trimming.
    :param rev_r_trim: True if `record_r` is reversed for trimming.
    :param coord_arr: Coordinate array containing all records.
    :param is_filter_fail_arr: True for all records with a non-PASS filter.
    :param match_qry: Match the query coordinates if `True`, otherwise, match the reference coordinates.
    :param is_rev_arr: "IS_REV" array for all records.
    :param score_model: Alignment model object or a configuration string describing a score model.
    :param op_arr_l: Alignment operation array for record at `index_l_trim`.
    :param op_arr_r: Alignment operation array for record at `index_r_trim`.

    :return: Tuple of trimmed alignment operation arrays for `record_l` and `record_r`.
    """

    filter_l = is_filter_fail_arr[index_l_trim]
    filter_r = is_filter_fail_arr[index_r_trim]

    record_l = coord_arr[index_l_trim]
    record_r = coord_arr[index_r_trim]

    # Orient operations so regions to be trimmed are at the head of the list
    if rev_l_trim:
        op_arr_l_mod = op_arr_l[::-1].copy()
    else:
        op_arr_l_mod = op_arr_l.copy()

    if rev_r_trim:
        op_arr_r_mod = op_arr_r[::-1].copy()
    else:
        op_arr_r_mod = op_arr_r.copy()

    # Get number of bases to trim. Assumes records overlap.
    if match_qry:

        diff_bp = np.min([record_l[COORD_QRY_END], record_r[COORD_QRY_END]]) - np.max([record_l[COORD_QRY_POS], record_r[COORD_QRY_POS]])

        if diff_bp < 0:
            raise RuntimeError('Records do not overlap in query space')

    else:
        if record_l[COORD_POS] > record_r[COORD_POS]:
            raise RuntimeError(f'Query sequences are incorrectly ordered in reference space')

        diff_bp = record_l[COORD_END] - record_r[COORD_POS]

        if diff_bp <= 0:
            raise RuntimeError(f'Records do not overlap in reference space')

    # Find the number of upstream (l) bases to trim to get to 0 (or query start)
    trace_l = trace_op_to_zero(op_arr_l_mod, diff_bp, match_qry, score_model)

    # Find the number of downstream (r) bases to trim to get to 0 (or query start)
    trace_r = trace_op_to_zero(op_arr_r_mod, diff_bp,  match_qry, score_model)

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
        raise RuntimeError(
            f'Program bug: Found no cut-sites'
        )

    # Get cut records
    cut_l = trace_l[cut_idx_l]
    cut_r = trace_r[cut_idx_r]

    # Set mid-record cuts (Left-align cuts, mismatch first, preferentially trim filtered records)
    residual_bp = diff_bp - (cut_l[TC_DIFF_CUM] + cut_r[TC_DIFF_CUM])

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

    # Get cut CIGAR String
    op_arr_l_mod = op_arr_l_mod[cut_l[TC_INDEX]:]
    op_arr_r_mod = op_arr_r_mod[cut_r[TC_INDEX]:]

    # Shorten last alignment record if set.
    op_arr_l_mod[0, 1] -= trim_l
    op_arr_r_mod[0, 1] -= trim_r

    # Modify alignment records
    cut_ref_l = cut_l[TC_REF_BP] + trim_l
    cut_qry_l = cut_l[TC_QRY_BP] + trim_l

    cut_ref_r = cut_r[TC_REF_BP] + trim_r
    cut_qry_r = cut_r[TC_QRY_BP] + trim_r

    if rev_l_trim:
        coord_arr[index_l_trim, COORD_END] -= cut_ref_l

        # Adjust positions in query space
        if is_rev_arr[index_l_trim]:
            coord_arr[index_l_trim, COORD_QRY_POS] += cut_qry_l
        else:
            coord_arr[index_l_trim, COORD_QRY_END] -= cut_qry_l

        # Track cut bases
        coord_arr[index_l_trim, COORD_TRIM_REF_R] += cut_ref_l
        coord_arr[index_l_trim, COORD_TRIM_QRY_R] += cut_qry_l

    else:
        coord_arr[index_l_trim, COORD_POS] += cut_ref_l

        # Adjust positions in query space
        if is_rev_arr[index_l_trim]:
            coord_arr[index_l_trim, COORD_QRY_END] -= cut_qry_l
        else:
            coord_arr[index_l_trim, COORD_QRY_POS] += cut_qry_l

        # Track cut bases
        coord_arr[index_l_trim, COORD_TRIM_REF_L] += cut_ref_l
        coord_arr[index_l_trim, COORD_TRIM_QRY_L] += cut_qry_l

    if rev_r_trim:
        coord_arr[index_r_trim, COORD_END] -= cut_ref_r

        # Adjust positions in query space
        if is_rev_arr[index_r_trim]:
            coord_arr[index_r_trim, COORD_QRY_POS] += cut_qry_r
        else:
            coord_arr[index_r_trim, COORD_QRY_END] -= cut_qry_r

        # Track cut bases
        coord_arr[index_r_trim, COORD_TRIM_REF_R] += cut_ref_r
        coord_arr[index_r_trim, COORD_TRIM_QRY_R] += cut_qry_r

    else:
        coord_arr[index_r_trim, COORD_POS] += cut_ref_r

        # Adjust positions in query space
        if is_rev_arr[index_r_trim]:
            coord_arr[index_r_trim, COORD_QRY_END] -= cut_qry_r
        else:
            coord_arr[index_r_trim, COORD_QRY_POS] += cut_qry_r

        # Track cut bases
        coord_arr[index_r_trim, COORD_TRIM_REF_L] += cut_ref_r
        coord_arr[index_r_trim, COORD_TRIM_QRY_L] += cut_qry_r

    # Add clipped bases to operations
    clip_l = cut_l[TC_CLIP_BP] + cut_l[TC_QRY_BP] + trim_l
    clip_r = cut_r[TC_CLIP_BP] + cut_r[TC_QRY_BP] + trim_r

    if clip_l > 0:
        op_arr_l_mod = np.append([(op.H, clip_l)], op_arr_l_mod, axis=0)

    if clip_r > 0:
        op_arr_r_mod = np.append([(op.H, clip_r)], op_arr_r_mod, axis=0)

    # Finish CIGAR and update score
    if rev_l_trim:
        op_arr_l_mod = op_arr_l_mod[::-1]

    if rev_r_trim:
        op_arr_r_mod = op_arr_r_mod[::-1]

    # Return trimmed records
    return op_arr_l_mod, op_arr_r_mod


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

    max_score = -np.inf      # Minimum cumulative score that may be cut
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

            # Save max
            if (
                score_cum > max_score or (  # Better event count, or
                    score_cum == max_score and (  # Same event count, and
                        max_diff_optimal is None or diff_optimal < max_diff_optimal  # Optimal difference is closer to 0 (less over-cut)
                    )
                )
            ):
                cut_idx_l = tc_idx_l
                cut_idx_r = tc_idx_r
                max_score = score_cum
                max_diff_optimal = diff_optimal

            tc_idx_r += 1

        # Reset right index
        tc_idx_r = tc_idx_r_start

    return cut_idx_l, cut_idx_r


def trace_op_to_zero(
        op_arr: np.ndarray,
        diff_bp: int,
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
        * TC_REF_BP = 5: Cumulative number of reference bases consumed up to this event, but not including it.
        * TC_QRY_BP = 6: Cumulative number of query bases consumed up to this event, but not including it.
        * TC_CLIP_BP = 7: Cumulative number of clipped bases (soft or hard). Alignments are not cut on clipped records,
            so cumulative and including does not affect the algorithm.
        * TC_SCORE_CUM = 8: Cumulative alignment score up to this event, but not including it.

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
    score_cumulative = 0

    ref_bp_sum = 0
    qry_bp_sum = 0
    clip_sum = 0

    trace_list = list()

    last_no_match = False  # Continue until the last element is a match

    while index < index_end and (diff_cumulative <= diff_bp or last_no_match or len(trace_list) == 0):
        op_count += 1
        op_code, op_len = op_arr[index]

        last_no_match = True

        if op_code == op.EQ:
            ref_bp = op_len
            qry_bp = op_len

            last_no_match = False

        elif op_code == op.X:
            ref_bp = op_len
            qry_bp = op_len

        elif op_code == op.I:
            ref_bp = 0
            qry_bp = op_len

        elif op_code == op.D:
            ref_bp = op_len
            qry_bp = 0

        elif op_code in op.CLIP_SET:
            ref_bp = 0
            qry_bp = 0

            clip_sum += op_len

        else:
            raise RuntimeError(
                'Illegal alignment operation while trimming alignment '
                f'(Op code #{op_code} at CIGAR index {index}): Expected op in "IDSH=X"'
            )

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
                    ref_bp_sum, qry_bp_sum,
                    clip_sum,
                    score_cumulative
                )
            )

        # Increment cumulative counts
        diff_cumulative += diff_change
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

def filter_depth(
        record_index: np.ndarray,
        coord_arr: np.ndarray,
        is_qry: bool,
        max_depth: int=20,
        max_overlap: int=0.8
) -> np.ndarray:
    """
    Determine which records pass this depth filter.

    :param record_index: List of record indices in `coord_array` to check. All indicies must be in the same reference or
        query sequence ID (i.e. POS/END or QRY_POS/QRY_END coordinates are assumed to be in the same sequence).
    :param coord_arr: Coordinate array for the whole alignment table.
    :param is_qry: Check depth in query coordinates if True, reference coordinates if False.
    :param max_depth: Filter records intersecting loci where alignments exceed this depth.
    :param max_overlap: Maximum overlap allowed when intersecting deep alignment regions. If the overlap is less than
        this value, do not filter. This allows long alignments to pass through regions with deep redundant alignments.
        If the total number of aligned bases divided by the total number of bases in deeply aligned regions is less
        than this overlap value, then do not filter.


    :return: Array of record indices passing the depth filter.
    """

    if is_qry:
        # Query coordinates
        coord_pos = COORD_QRY_POS
        coord_end = COORD_QRY_END

    else:
        # Reference coordinates
        coord_pos = COORD_POS
        coord_end = COORD_END

    # Three column array:
    # 1) Coordinates where an alignment begins or ends
    # 2) 1 for alignment start, -1 for alignment end
    # 3) Index from record_index
    depth_edge_arr = np.asarray(
        [
            np.concatenate([coord_arr[record_index, coord_pos], coord_arr[record_index, coord_end]]),
            np.concatenate([np.ones(record_index.shape[0]), -np.ones(record_index.shape[0])]),
            np.concatenate([record_index, record_index])
        ]
    ).astype(int).T

    # Sort by coordinates (first column)
    depth_edge_arr = depth_edge_arr[depth_edge_arr[:, 0].argsort()]

    # Find contiguous ranges where depth is exceeded
    mask = np.cumsum(depth_edge_arr[:, 1]) > max_depth

    if not np.any(mask):
        return record_index

    changes = np.diff(mask.astype(int), prepend=0)

    filter_coords = np.asarray(
        [
            depth_edge_arr[np.where(changes == 1)[0], 0].astype(int),
            depth_edge_arr[np.where(changes == -1)[0], 0].astype(int)
        ]
    ).T

    # Create an index mask
    filter_arr = np.ones(record_index.shape[0], dtype=bool)

    for iter_index in range(record_index.shape[0]):
        index = record_index[iter_index]

        overlap = np.minimum(
            filter_coords[:, 1], coord_arr[index, coord_end]
        ) - np.maximum(
            filter_coords[:, 0], coord_arr[index, coord_pos]
        )

        if (
            np.sum(np.where(overlap > 0, overlap, 0))
        ) / (
            coord_arr[index, coord_end] - coord_arr[index, coord_pos]
        ) > max_overlap:
            filter_arr[iter_index] = False

    # Subset indices
    return record_index[filter_arr]


    # Find a set of all record indices where depth exceeds max_depth
    # return set(
    #     depth_edge_arr[
    #         np.where(np.cumsum(depth_edge_arr[:, 1]) > max_depth),  # Cumulative sum over second column
    #         2  # Index column
    #     ][0].astype(int)
    # )

def check_trim_qry(df):
    """
    Check for alignment overlaps in query coordinates and raise an exception if found.

    :param df: Alignment dataframe after query trimming.
    """

    itree = collections.defaultdict(intervaltree.IntervalTree)

    index_tuple_set = set()

    for index, row in df.iterrows():
        for match in itree[row['QRY_ID']][row['QRY_POS']:row['QRY_END']]:
            index_tuple_set.add((row['INDEX'], match.data))

        itree[row['QRY_ID']][row['QRY_POS']:row['QRY_END']] = row['INDEX']

    if index_tuple_set:
        n = len(index_tuple_set)
        index_list = ', '.join([f'{i1}-{i2}' for i1, i2 in sorted(index_tuple_set)[:3]]) + ('...' if n > 3 else '')

        raise RuntimeError(f'Found {n} overlapping query alignments: INDEX pairs {index_list}')

def check_trim_ref(df):
    """
    Check for alignment overlaps in query coordinates and raise an exception if found.

    :param df: Alignment dataframe after query trimming.
    """

    itree = collections.defaultdict(intervaltree.IntervalTree)

    index_tuple_set = set()

    for index, row in df.iterrows():
        for match in itree[row['#CHROM']][row['POS']:row['END']]:
            index_tuple_set.add((row['INDEX'], match.data))

        itree[row['#CHROM']][row['POS']:row['END']] = row['INDEX']

    if index_tuple_set:
        n = len(index_tuple_set)
        index_list = ', '.join([f'{i1}-{i2}' for i1, i2 in sorted(index_tuple_set)[:3]]) + ('...' if n > 3 else '')

        raise RuntimeError(f'Found {n} overlapping reference alignments: INDEX pairs {index_list}')
