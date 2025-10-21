"""Functions for trimming alignments.

In repeat-mediated events, aligners may align the same parts of a query
sequence to both reference copies (e.g. large DEL) or two parts of a query
sequence to the same region (e.g. tandem duplication). This function trims
back the alignments using the CIGAR string until the overlap is resolved
using a simple greedy algorithm that maximizes the number of variants removed
from the alignment during trimming (each variant is an insertion, deletion,
or SNVs; currently, no bonus is given to removing larger insertions or
deletions vs smaller ones).

For example, a large repeat-mediated deletion will have two reference copies,
but one copy in the query, and the single query copy is aligned to both by
breaking the alignment record into two (one up to the deletion, and one
following it). If the query coordinates were ignored, the alignment gap is
smaller than the actual deletion event and one or both sides of the deletion
are filled with false variants. In this example, the alignment is walked-
out from both ends of the deletion until there is no duplication of aligned
query (e.g. the alignment stops at one query base and picks up at the next
query base). In this case, this function would be asked to resolve the
query coordinates (match_qry = True).

A similar situation occurs for large tandem duplications, except there is one
copy in the reference and two (or more) in the query. Aligners may align
through the reference copy, break the alignment, and start a new alignment
through the second copy in the query. In this case, this function would be
asked to resolve reference coordinates (match_qry = False).
"""

__all__ = [
    'TRIM_DESC',
    'DEFAULT_MIN_TRIM_QRY_LEN',
    'trim_alignments_qry',
    'trim_alignments_ref',
    'check_overlap_qry',
    'check_overlap_ref',
]


from typing import Any, Optional

import numpy as np
import polars as pl

from . import op

from .features import FeatureGenerator
from .score import ScoreModel, get_score_model
from .tables import qry_order_expr, check_table

# Indices for tuples returned by trace_op_to_zero()
_TC_INDEX = 0
"""Index in the operation array in tuples returned by trace_op_to_zero()."""

_TC_OP_CODE = 1
"""Operation code in tuples returned by trace_op_to_zero()."""

_TC_OP_LEN = 2
"""Operation length in tuples returned by trace_op_to_zero()."""

_TC_DIFF_CUM = 3
"""Cumulative base difference up this event (excluding this event) in tuples returned by trace_op_to_zero()."""

_TC_DIFF = 4
"""Base difference for this event (op_len depending on op_code) in tuples returned by trace_op_to_zero()."""

_TC_REF_BP = 5
"""Cumulative reference bases up to this event (excluding this event) in tuples returned by trace_op_to_zero()."""

_TC_QRY_BP = 6
"""Cumulative query bases up to this event (excluding this event) in tuples returned by trace_op_to_zero()."""

_TC_CLIP_BP = 7
"""Cumulative clipped bases (soft or hard) in tuples returned by trace_op_to_zero()."""

_TC_SCORE_CUM = 8
"""Cumulative alignment score up to this event (excluding this event) in tuples returned by trace_op_to_zero()."""

TRIM_DESC = {
    'none': 'No trimming',
    'qry': 'Query-only trimming',
    'qryref': 'Query-Reference trimming'
}
"""Trim mode description."""

DEFAULT_MIN_TRIM_QRY_LEN = 500
"""Default minimum query length. Alignments with fewer query bases are removed."""


def trim_alignments_qry(
        df: pl.DataFrame | pl.LazyFrame,
        df_qry_fai: pl.DataFrame,
        score_model: Optional[ScoreModel | str] = None,
        min_trim_qry_len: int = DEFAULT_MIN_TRIM_QRY_LEN,
):
    """Trim alignments to remove query bases appearing in multiple alignments.

    After trimming, every query base appears in one or zero alignments.

    :param df: Alignment dataframe.
    :param df_qry_fai: Query FAI dataframe.
    :param score_model: Alignment score model.
    :param min_trim_qry_len: Minimum query length to trim.

    :returns: A table of query-trimmed alignments.
    """
    if df is None:
        raise ValueError('Alignment dataframe is None.')

    if df_qry_fai is None:
        raise ValueError('Query FAI dataframe is None.')

    if min_trim_qry_len < 1:
        raise ValueError('min_trim_qry_len must be at least 1.')

    score_model = get_score_model(score_model)

    # Columns to drop at the end
    drop_cols = []

    # Create feature generator
    feature_gen = FeatureGenerator(
        features=None,
        score_model=score_model,
        force_all=True  # Not necessary, but overwrite features if already present
    )

    # Prepare alignment table
    if 'score' not in df.columns:
        df = FeatureGenerator(
            features=('score',), score_model=score_model
        )(
            df=df, df_qry_fai=df_qry_fai
        )
        drop_cols.append('score')
    elif (n_null := df.select(pl.col('score').is_null().sum()).item()) > 0:
        raise ValueError(f'Alignment score is missing for {n_null} alignment records.')

    # Prepare table features
    if not isinstance(df, pl.LazyFrame):
        df = df.lazy()

    df = (
        df
        .drop('_index', '_active', strict=False)
        .with_row_index('_index')
        .filter(
            pl.col('qry_end') - pl.col('qry_pos') >= min_trim_qry_len
        )
        .with_columns(
            (pl.col('filter').list.len() == 0).alias('_is_filtered')
        )
    )

    drop_cols.append('_index')
    drop_cols.append('_is_filtered')

    # Get unique query IDs

    qry_id_list = df.select(pl.col('qry_id').unique().sort()).collect().to_series().to_list()

    subtable_list = list()

    subtable_schema = list(df.collect_schema().items()) + [('_active', pl.Boolean)]

    # Mechanism for caching rows
    row_cache: dict[int, dict[str, Any]] = dict()

    def get_row(index: int) -> dict[str, Any]:
        if index not in row_cache:
            row_cache[index] = df.filter(pl.col('_index') == index).collect().rows(named=True)[0]
            row_cache[index]['_active'] = True
        return row_cache[index]

    for qry_id in qry_id_list:
        row_cache.clear()

        # Get overlaps
        df_overlap = (
            df.filter(pl.col('qry_id') == qry_id)
            .join_where(
                df.filter(pl.col('qry_id') == qry_id),
                pl.col('_index') != pl.col('_index_right'),
                pl.col('qry_pos') < pl.col('qry_end_right'),
                pl.col('qry_end') > pl.col('qry_pos_right'),
                suffix='_right'
            )
            .with_columns(
                pl.min_horizontal([pl.col('score'), pl.col('score_right')]).alias('score_min'),
                pl.max_horizontal([pl.col('score'), pl.col('score_right')]).alias('score_max')
            )
            .sort([
                'score_max', 'score_min',
                'qry_pos', 'qry_pos_right',
                'qry_end', 'qry_end_right',
                '_index', '_index_right'
            ])
            .select(['_index', '_index_right'])
            .collect()
        )

        # Process overlaps
        for index_l, index_r in df_overlap.rows():
            row_l = get_row(index_l)
            row_r = get_row(index_r)

            # Check records - overlap & active flags might change through rounds of trimming
            if not row_l['_active'] or not row_r['_active']:
                # One record is not active
                continue

            if row_l['qry_pos'] >= row_r['qry_end'] or row_l['qry_end'] <= row_r['qry_pos']:
                # No overlap
                continue

            if (
                (row_l['qry_pos'] - row_r['qry_pos'])
            ) * (
                (row_r['qry_end'] - row_l['qry_end'])
            ) >= 0:
                # One record contained within the other: Set the least inactive (by score, then by length)
                sorted([
                    (row_l['score'], row_l['qry_end'] - row_l['qry_pos'], row_l['_index'], row_l),
                    (row_r['score'], row_r['qry_end'] - row_r['qry_pos'], row_r['_index'], row_r)
                ])[0][-1]['_active'] = False

                continue

            # Order records so the correct end is trimmed (right end of row_l is trimmed)
            if row_l['qry_pos'] > row_r['qry_pos']:
                row_l, row_r = row_r, row_l

            # Determine trim orientation (right side of index_l is to be trimmed, must be reversed so
            # trimmed alignment records are at the beginning; left side of index_r is to be trimmed, which is
            # already at the start of the alignment operation list).
            rev_l = not row_l['is_rev']  # Trim right end of index_l
            rev_r = row_r['is_rev']      # Trim left end of index_r

            # Determine which side to preferentially trim in case of ties
            if row_l['chrom'] == row_r['chrom'] and row_l['is_rev'] != row_r['is_rev']:
                # Same chromosome and orientation, preferentially trim left-most record to maintain left-aligning
                # breakpoints.

                prefer_l = (
                    row_l['pos'] if row_l['is_rev'] else row_l['end']  # Reference pos on left record being trimmed
                ) <= (
                    row_r['end'] if row_r['is_rev'] else row_r['pos']  # Reference pos on right record being trimmed
                )

            else:
                prefer_l = row_l['score'] < row_r['score']

            # Run trimming
            try:
                if prefer_l:
                    _trim_alignment_record(
                        row_l=row_l,
                        row_r=row_r,
                        rev_l=rev_l,
                        rev_r=rev_r,
                        match_qry=True,
                        score_model=score_model
                    )
                else:
                    _trim_alignment_record(
                        row_l=row_r,
                        row_r=row_l,
                        rev_l=rev_r,
                        rev_r=rev_l,
                        match_qry=True,
                        score_model=score_model,
                    )
            except Exception as e:
                raise ValueError(
                    (
                        'Error trimming overlapping alignments in query coordinates: '
                        '[side=L, align_index={align_index_l}, ref=({chrom_l}, {pos_l:,}, {end_l:,}) '
                        'qry=({qry_id_l}, {qry_pos_l:,}, {qry_end_l:,})]; '
                        '[side=R, align_index={align_index_r}, ref=({chrom_r}, {pos_r:,}, {end_r:,}) '
                        'qry=({qry_id_r}, {qry_pos_r:,}, {qry_end_r:,})]: '
                        '{e}'
                    ).format(
                        **(
                                {key + '_l': val for key, val in row_l.items()} |
                                {key + '_r': val for key, val in row_r.items()} |
                                {'e': e}
                        )
                    )
                )

            # Check trimming
            if row_r['qry_pos'] < row_l['qry_end']:
                raise ValueError(
                    (
                        'Found overlapping query bases after trimming in query coordinates: '
                        '[side=L, align_index={align_index_l}, ref=({chrom_l}, {pos_l:,}, {end_l:,}) '
                        'qry=({qry_id_l}, {qry_pos_l:,}, {qry_end_l:,})]; '
                        '[side=R, align_index={align_index_r}, ref=({chrom_r}, {pos_r:,}, {end_r:,}) '
                        'qry=({qry_id_r}, {qry_pos_r:,}, {qry_end_r:,})]'
                    ).format(
                        **(
                                {key + '_l': val for key, val in row_l.items()} |
                                {key + '_r': val for key, val in row_r.items()}
                        )
                    )
                )

            # Modify if new aligned size is at least min_trim_qry_len, remove if shorter
            if row_l['qry_end'] - row_l['qry_pos'] < min_trim_qry_len:
                row_l['_active'] = False

            if row_r['qry_end'] - row_r['qry_pos'] < min_trim_qry_len:
                row_r['_active'] = False

        # Add to sub-table: Unmodified records, modified records (drop eliminated records)
        df_qry_mod = feature_gen(
            df=(
                pl.from_dicts(
                    list(row_cache.values()),
                    schema=subtable_schema
                )
            ),
            df_qry_fai=df_qry_fai
        ).lazy()

        subtable_list.append(
            pl.concat([
                feature_gen(
                    df=(
                        df_qry_mod
                        .filter(pl.col('_active'))
                        .drop('_active')
                        .collect()
                    ),
                    df_qry_fai=df_qry_fai
                ),
                (
                    df.filter(
                        pl.col('qry_id') == qry_id,
                    )
                    .join(
                        df_qry_mod,
                        on='_index',
                        how='anti'
                    )
                    .collect()
                )
            ])
        )

    df_trim = (  # Concat and update features
        pl.concat(subtable_list)
        .sort('_index')
        .with_columns(qry_order_expr())
        .drop(drop_cols)
    )

    check_table(df_trim)

    return df_trim


def trim_alignments_ref(
        df: pl.DataFrame | pl.LazyFrame,
        df_qry_fai: pl.DataFrame,
        score_model: Optional[ScoreModel | str] = None,
        min_trim_qry_len: int = DEFAULT_MIN_TRIM_QRY_LEN,
        on_qry: bool = False
):
    """Trim alignments to remove reference bases appearing in multiple alignments.

    After trimming, every reference base appears in one or zero alignments.

    :param df: Alignment dataframe.
    :param df_qry_fai: Query FAI dataframe.
    :param score_model: Alignment score model.
    :param min_trim_qry_len: Minimum query length to trim.
    :param on_qry: If True, only trim alignments where the query ID matches. This allows for redundant alignments
        in a single reference locus, for example, multiple haplotypes present in the assembly because they were
        not phased or because of anneuploidy. If this option is True, downstream filtering should be applied to
        the callset to clean up the callset.

    :returns: A table of query-trimmed alignments.
    """
    if df is None:
        raise ValueError('Alignment dataframe is None.')

    if df_qry_fai is None:
        raise ValueError('Query FAI dataframe is None.')

    if min_trim_qry_len < 1:
        raise ValueError('min_trim_qry_len must be at least 1.')

    score_model = get_score_model(score_model)

    # Columns to drop at the end
    drop_cols = []

    # Create feature generator
    feature_gen = FeatureGenerator(
        features=None,
        score_model=score_model,
        force_all=True  # Not necessary, but overwrite features if already present
    )

    # Prepare alignment table
    if 'score' not in df.columns:
        df = FeatureGenerator(
            features=('score',), score_model=score_model
        )(
            df=df, df_qry_fai=df_qry_fai
        )
        drop_cols.append('score')
    elif (n_null := df.select(pl.col('score').is_null().sum()).item()) > 0:
        raise ValueError(f'Alignment score is missing for {n_null} alignment records.')

    # Prepare table features
    if not isinstance(df, pl.LazyFrame):
        df = df.lazy()

    df = (
        df
        .drop('_index', '_active', strict=False)
        .with_row_index('_index')
        .filter(
            pl.col('qry_end') - pl.col('qry_pos') >= min_trim_qry_len
        )
        .with_columns(
            (pl.col('filter').list.len() == 0).alias('_is_filtered')
        )
    )

    drop_cols.append('_index')
    drop_cols.append('_is_filtered')

    chrom_list = df.select(pl.col('chrom').unique().sort()).collect().to_series().to_list()

    subtable_list = list()

    subtable_schema = list(df.collect_schema().items()) + [('_active', pl.Boolean)]

    # Mechanism for caching rows (moved outside loop to avoid closure issues)
    row_cache: dict[int, dict[str, Any]] = dict()

    def get_row(index: int) -> dict[str, Any]:
        if index not in row_cache:
            row_cache[index] = df.filter(pl.col('_index') == index).collect().rows(named=True)[0]
            row_cache[index]['_active'] = True
        return row_cache[index]

    for chrom in chrom_list:
        row_cache.clear()

        # Get overlaps
        df_overlap = (
            df.filter(pl.col('chrom') == chrom)
            .join_where(
                df.filter(pl.col('chrom') == chrom),
                pl.col('_index') != pl.col('_index_right'),
                pl.col('pos') < pl.col('end_right'),
                pl.col('end') > pl.col('pos_right'),
                suffix='_right'
            )
            .with_columns(
                pl.min_horizontal([pl.col('score'), pl.col('score_right')]).alias('score_min'),
                pl.max_horizontal([pl.col('score'), pl.col('score_right')]).alias('score_max')
            )
            .sort([
                'score_max', 'score_min',
                'pos', 'pos_right',
                'end', 'end_right',
                '_index', '_index_right'
            ])
            .select(['_index', '_index_right'])
            .collect()
        )

        # Process overlaps
        for index_l, index_r in df_overlap.rows():
            row_l = get_row(index_l)
            row_r = get_row(index_r)

            # Check records - overlap & active flags might change through rounds of trimming
            if not row_l['_active'] or not row_r['_active']:
                # One record is not active
                continue

            if row_l['pos'] >= row_r['end'] or row_l['end'] <= row_r['pos']:
                # No overlap
                continue

            if on_qry and row_l['qry_id'] != row_r['qry_id']:
                # Query IDs do not match and trimming is set to only query context
                continue

            if (
                (row_l['pos'] - row_r['pos'])
            ) * (
                (row_r['end'] - row_l['end'])
            ) >= 0:
                # One record contained within the other: Set the least inactive (by score, then by length)
                sorted([
                    (row_l['score'], row_l['end'] - row_l['pos'], row_l['_index'], row_l),
                    (row_r['score'], row_r['end'] - row_r['pos'], row_r['_index'], row_r)
                ])[0][-1]['_active'] = False

                continue

            # Order records so the correct end is trimmed (right end of row_l is trimmed)
            if row_l['pos'] > row_r['pos']:
                row_l, row_r = row_r, row_l

            # Determine trim orientation (right side of index_l is to be trimmed, must be reversed so
            # trimmed alignment records are at the beginning; left side of index_r is to be trimmed, which is
            # already at the start of the alignment operation list).
            try:
                _trim_alignment_record(
                    row_l=row_l,
                    row_r=row_r,
                    rev_l=True,
                    rev_r=False,
                    match_qry=False,
                    score_model=score_model
                )

            except Exception as e:
                raise ValueError(
                    (
                        'Error trimming overlapping alignments in reference coordinates: '
                        '[side=L, align_index={align_index_l}, ref=({chrom_l}, {pos_l:,}, {end_l:,}) '
                        'qry=({qry_id_l}, {qry_pos_l:,}, {qry_end_l:,})]; '
                        '[side=R, align_index={align_index_r}, ref=({chrom_r}, {pos_r:,}, {end_r:,}) '
                        'qry=({qry_id_r}, {qry_pos_r:,}, {qry_end_r:,})]: '
                        '{e}'
                    ).format(
                        **(
                                {key + '_l': val for key, val in row_l.items()} |
                                {key + '_r': val for key, val in row_r.items()} |
                                {'e': e}
                        )
                    )
                )

            # Check trimming
            if row_r['pos'] < row_l['end']:
                raise ValueError(
                    (
                        'Found overlapping query bases after trimming in reference coordinates: '
                        '[side=L, align_index={align_index_l}, ref=({chrom_l}, {pos_l:,}, {end_l:,}) '
                        'qry=({qry_id_l}, {qry_pos_l:,}, {qry_end_l:,})]; '
                        '[side=R, align_index={align_index_r}, ref=({chrom_r}, {pos_r:,}, {end_r:,}) '
                        'qry=({qry_id_r}, {qry_pos_r:,}, {qry_end_r:,})]'
                    ).format(
                        **(
                                {key + '_l': val for key, val in row_l.items()} |
                                {key + '_r': val for key, val in row_r.items()}
                        )
                    )
                )

            # Modify if new aligned size is at least min_trim_qry_len, remove if shorter
            if row_l['qry_end'] - row_l['qry_pos'] < min_trim_qry_len:
                row_l['_active'] = False

            if row_r['qry_end'] - row_r['qry_pos'] < min_trim_qry_len:
                row_r['_active'] = False

        # Add to sub-table: Unmodified records, modified records (drop eliminated records)
        df_qry_mod = feature_gen(
            df=(
                pl.from_dicts(
                    list(row_cache.values()),
                    schema=subtable_schema
                )
            ),
            df_qry_fai=df_qry_fai
        ).lazy()

        subtable_list.append(
            pl.concat([
                feature_gen(
                    df=(
                        df_qry_mod
                        .filter(pl.col('_active'))
                        .drop('_active')
                        .collect()
                    ),
                    df_qry_fai=df_qry_fai
                ),
                (
                    df.filter(
                        pl.col('chrom') == chrom,
                    )
                    .join(
                        df_qry_mod,
                        on='_index',
                        how='anti'
                    )
                    .collect()
                )
            ])
        )

    df_trim = (  # Concat and update features
            pl.concat(subtable_list)
            .sort('_index')
            .with_columns(qry_order_expr())
            .drop(drop_cols)
    )

    check_table(df_trim)

    return df_trim


def _trim_alignment_record(
        row_l: dict[str, Any],
        row_r: dict[str, Any],
        rev_l: bool,
        rev_r: bool,
        match_qry: bool,
        score_model: ScoreModel
):
    """Trim ends of overlapping alignments.

    Remove overlapping ends modify records (row_l and row_r) in place.

    :param row_l: Left alignment record.
    :param row_r: Right alignment record.
    :param rev_l: True if left alignment record is reversed.
    :param rev_r: True if right alignment record is reversed.
    :param match_qry: True if query coordinates should be matched exactly.
    :param score_model: Alignment score model.
    """
    # Get operation arrays
    op_arr_l = op.row_to_arr(row_l)
    op_arr_r = op.row_to_arr(row_r)

    # Orient operations so regions to be trimmed are at the head of the list
    if rev_l:
        op_arr_l = op_arr_l[::-1]

    if rev_r:
        op_arr_r = op_arr_r[::-1]

    # Get number of bases to trim. Assumes records overlap.
    if match_qry:
        diff_bp = min([row_l['qry_end'], row_r['qry_end']]) - max(row_l['qry_pos'], row_r['qry_pos'])

        if diff_bp < 0:
            raise RuntimeError('Records do not overlap in query space')

    else:
        if row_l['pos'] > row_r['pos']:
            raise RuntimeError(
                f'Query sequences are incorrectly ordered in reference space: '
                f'row_l["pos"] ({row_l["pos"]}) > row_r["pos"] ({row_r["pos"]})'
            )

        diff_bp = row_l['end'] - row_r['pos']

        if diff_bp <= 0:
            raise RuntimeError('Records do not overlap in reference space')

    # Find the number of upstream (l) bases to trim to get to 0 (or query start)
    trace_l = _trace_op_to_zero(op_arr_l, diff_bp, match_qry, score_model)

    # Find the number of downstream (r) bases to trim to get to 0 (or query start)
    trace_r = _trace_op_to_zero(op_arr_r, diff_bp, match_qry, score_model)

    # For each upstream alignment cut-site, find the best matching downstream alignment cut-site. Not all cut-site
    # combinations need to be tested since trimmed bases and event count is non-decreasing as it moves away from the
    # best cut-site (residual overlapping bases 0 and maximum events consumed)
    if row_l['_is_filtered'] == row_r['_is_filtered']:
        # Find optimal cut sites.
        # cut_idx_l and cut_idx_r are indices to trace_l and trace_r. These trace records point to the last alignment
        # operation to survive the cut, although they may be truncated (e.g. 100= to 90=).
        cut_idx_l, cut_idx_r = _find_cut_sites(trace_l, trace_r, diff_bp, score_model)

    else:
        # One alignment record is filtered, but the other is not. Preferentially truncate the filtered alignment.
        if row_l['_is_filtered']:
            cut_idx_l = len(trace_l) - 1
            cut_idx_r = 0
        else:
            cut_idx_l = 0
            cut_idx_r = len(trace_r) - 1

    # Check for no cut-sites. Should not occur at this stage.
    assert cut_idx_l is not None and cut_idx_r is not None, 'No cut sites found in overlapping alignments'

    # Get cut records
    cut_l = trace_l[cut_idx_l]
    cut_r = trace_r[cut_idx_r]

    # Set mid-record cuts (Left-align cuts, mismatch first, preferentially trim filtered records)
    residual_bp = diff_bp - (cut_l[_TC_DIFF_CUM] + cut_r[_TC_DIFF_CUM])

    trim_dict = {'l': 0, 'r': 0}
    cut_dict = {'l': cut_l, 'r': cut_r}

    trim_order = {
        (False, False): [('l', op.X), ('r', op.X), ('l', op.EQ), ('r', op.EQ)],
        (True, True): [('l', op.X), ('r', op.X), ('l', op.EQ), ('r', op.EQ)],
        (True, False): [('l', op.X), ('l', op.EQ), ('r', op.X), ('r', op.EQ)],
        (False, True): [('r', op.X), ('r', op.EQ), ('l', op.X), ('l', op.EQ)],
    }

    for side, op_code in trim_order[(row_l['_is_filtered'], row_r['_is_filtered'])]:
        if residual_bp > 0 and cut_dict[side][_TC_OP_CODE] == op_code:
            trim_bp = int(np.min([residual_bp, cut_dict[side][_TC_OP_LEN] - 1]))
            trim_dict[side] += trim_bp
            residual_bp -= trim_bp

    trim_l = trim_dict['l']
    trim_r = trim_dict['r']

    # Get cut CIGAR String
    op_arr_l = op_arr_l[cut_l[_TC_INDEX]:]
    op_arr_r = op_arr_r[cut_r[_TC_INDEX]:]

    # Shorten last alignment record if set.
    op_arr_l[0, 1] -= trim_l
    op_arr_r[0, 1] -= trim_r

    # Modify alignment records
    cut_ref_l = cut_l[_TC_REF_BP] + trim_l
    cut_qry_l = cut_l[_TC_QRY_BP] + trim_l

    cut_ref_r = cut_r[_TC_REF_BP] + trim_r
    cut_qry_r = cut_r[_TC_QRY_BP] + trim_r

    if rev_l:
        row_l['end'] -= cut_ref_l

        # Adjust positions in query space
        if row_l['is_rev']:
            row_l['qry_pos'] += cut_qry_l
        else:
            row_l['qry_end'] -= cut_qry_l

    else:
        row_l['pos'] += cut_ref_l

        # Adjust positions in query space
        if row_l['is_rev']:
            row_l['qry_end'] -= cut_qry_l
        else:
            row_l['qry_pos'] += cut_qry_l

    if rev_r:
        row_r['end'] -= cut_ref_r

        # Adjust positions in query space
        if row_r['is_rev']:
            row_r['qry_pos'] += cut_qry_r
        else:
            row_r['qry_end'] -= cut_qry_r

    else:
        row_r['pos'] += cut_ref_r

        # Adjust positions in query space
        if row_r['is_rev']:
            row_r['qry_end'] -= cut_qry_r
        else:
            row_r['qry_pos'] += cut_qry_r

    # Add clipped bases to operations
    clip_l = cut_l[_TC_CLIP_BP] + cut_l[_TC_QRY_BP] + trim_l
    clip_r = cut_r[_TC_CLIP_BP] + cut_r[_TC_QRY_BP] + trim_r

    if clip_l > 0:
        op_arr_l = np.append([(op.H, clip_l)], op_arr_l, axis=0)

    if clip_r > 0:
        op_arr_r = np.append([(op.H, clip_r)], op_arr_r, axis=0)

    # Update operations
    if rev_l:
        op_arr_l = op_arr_l[::-1]

    if rev_r:
        op_arr_r = op_arr_r[::-1]

    row_l['align_ops'] = op.arr_to_row(op_arr_l)
    row_r['align_ops'] = op.arr_to_row(op_arr_r)


def _find_cut_sites(
        trace_l: list[tuple],
        trace_r: list[tuple],
        diff_bp: int,
        score_model: ScoreModel
):
    """Find best cut-sites for left and right alignments to consume `diff_bp` bases.

    Optimize by:

        #. `diff_bp` or more bases removed (avoid over-trimming)
        #. Minimize lost score (prefer dropping events with larger negative scores)
        #. Tie-break by:
            #. Total removed bases closest to `diff_bp`.
            #. Left-align break (trace_l is preferentially trimmed when there is a tie).

    :param trace_l: List of tuples for the left alignment generated by `trace_op_to_zero()`.
    :param trace_r: List of tuples for the right alignment generated by `trace_op_to_zero()`.
    :param diff_bp: Target removing this many bases. Derived from reference or query (depending on which is trimmed).
    :param score_model: Alignment score model.

    :returns: Tuple of (cut_idx_l, cut_idx_r). cut_idx_l and cut_idx_r are the left query and right query operation list
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

    # Optimal difference in the number of bases cut over diff_bp. closest to 0 means cut-site
    # can be placed exactly and does not force over-cutting to remove overlap)
    max_diff_optimal = None

    # Traverse l cut-sites
    for tc_idx_l in range(len(trace_l) - 1, -1, -1):

        # Get min and max base differences achievable by cutting at the end or beginning of this l-record.
        min_bp_l = trace_l[tc_idx_l][_TC_DIFF_CUM]
        max_bp_l = trace_l[tc_idx_l][_TC_DIFF_CUM] + trace_l[tc_idx_l][_TC_DIFF] - 1  # Cut all but one left base

        # Traverse r cut-sites until max-left + max-right base difference diff_bp or greater.
        while (
                tc_idx_r + 1 < len_r and

                # Cut all but one right base
                max_bp_l + trace_r[tc_idx_r][_TC_DIFF_CUM] + trace_r[tc_idx_r][_TC_DIFF] - 1 < diff_bp
        ):
            tc_idx_r += 1

        # Traverse all cases where max-cutting the left event crosses 0 residual bases
        # (or the single case resulting in over-cutting). After this loop, the range of
        # acceptable right indices spans tc_idx_r_start to tc_idx_r (exclusive on right side).
        tc_idx_r_start = tc_idx_r

        while (
                tc_idx_r < len_r and (
                min_bp_l + trace_r[tc_idx_r][_TC_DIFF_CUM] <= diff_bp or  # Acceptable site not found
                tc_idx_r == tc_idx_r_start  # Find at least one cut-site on the right side, even if it over-cuts.
                )
        ):

            # Collect cut-site stats
            max_bp = max_bp_l + trace_r[tc_idx_r][_TC_DIFF_CUM] + trace_r[tc_idx_r][_TC_DIFF] - 1

            diff_min = diff_bp - max_bp

            # Count number of events if the minimal cut at these sites are made.
            score_cum = trace_l[tc_idx_l][_TC_SCORE_CUM] + trace_r[tc_idx_r][_TC_SCORE_CUM]

            if diff_min <= 0:

                residual_x = np.min([
                    np.abs(diff_min),
                    (
                        trace_l[tc_idx_l][_TC_OP_LEN] - 1 if trace_l[tc_idx_l][_TC_OP_CODE] == op.X else 0
                    ) + (
                        trace_r[tc_idx_r][_TC_OP_LEN] - 1 if trace_r[tc_idx_r][_TC_OP_CODE] == op.X else 0
                    )
                ])

                diff_min += residual_x

                residual_eq = np.min([
                    np.abs(diff_min),
                    (
                        trace_l[tc_idx_l][_TC_OP_LEN] - 1 if trace_l[tc_idx_l][_TC_OP_CODE] == op.EQ else 0
                    ) + (
                        trace_r[tc_idx_r][_TC_OP_LEN] - 1 if trace_r[tc_idx_r][_TC_OP_CODE] == op.EQ else 0
                    )
                ])

                score_cum += score_model.score_op(op.X, residual_x) + score_model.score_op(op.EQ, residual_eq)

                diff_optimal = 0  # diff_bp is exactly achievable
            else:
                # Must over-cut to use these sites.
                diff_optimal = diff_min

            # Save max
            if (
                score_cum > max_score or (  # Better event count, or
                    score_cum == max_score and (  # Same event count, and
                        # Optimal difference is closer to 0 (less over-cut)
                        max_diff_optimal is None or diff_optimal < max_diff_optimal
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


def _trace_op_to_zero(
        op_arr: np.ndarray,
        diff_bp: int,
        diff_query: bool,
        score_model: ScoreModel
) -> list[tuple]:
    """Trace align operations back until diff_bp query bases are discarded from the alignment.

    Operations must only contain operators "IDSH=X" (no "M"). The array returned is only alignment match ("=" or "X"
    records) for the optimal-cut algorithm (can only cut at aligned bases).

    Returns a list of tuples for each operation traversed:

        * TC_INDEX = 0: Index in the operation array.
        * TC_OP_CODE = 1: operation code (character, e.g. "I", "=").
        * TC_OP_LEN = 2: Operation length.
        * TC_DIFF_CUM = 3: Cumulative base difference up this event, but not including it.
        * TC_DIFF = 4: Base difference for this event. Will be op_len depending on the operation code.
        * TC_REF_BP = 5: Cumulative number of reference bases consumed up to this event, but not including it.
        * TC_QRY_BP = 6: Cumulative number of query bases consumed up to this event, but not including it.
        * TC_CLIP_BP = 7: Cumulative number of clipped bases (soft or hard). Alignments are not cut on clipped records,
            so cumulative and including does not affect the algorithm.
        * TC_SCORE_CUM = 8: Cumulative alignment score up to this event, but not including it.

    :param op_arr: Array of alignment operations (col 1: op_code, col 2: op_len).
    :param diff_bp: Number of query bases to trace back. Final record will traverse past this value.
    :param diff_query: Compute base differences for query sequence if `True`. If `False`, compute for reference.
    :param score_model: Alignment score model.

    :returns: A list of tuples tracing the effects of truncating an alignment at a given alignment operation.
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
                    int(index),
                    int(op_code), int(op_len),
                    int(diff_cumulative), int(diff_change),
                    int(ref_bp_sum), int(qry_bp_sum),
                    int(clip_sum),
                    float(score_cumulative)
                )
            )

        # Increment cumulative counts
        diff_cumulative += diff_change
        score_cumulative += score_model.score_op(op_code, op_len)

        ref_bp_sum += ref_bp
        qry_bp_sum += qry_bp

        index += 1

    return trace_list


def check_overlap_qry(
        df: pl.DataFrame
):
    """Check for alignment overlaps in query coordinates and raise an exception if any are found.

    :param df: Alignment dataframe after query trimming.

    :raises ValueError: If overlaps in query coordinates are found.
    """
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    df = (
        df
        .drop('_index', strict=False)
        .with_row_index('_index')
    )

    qry_id_list = df.select(pl.col('qry_id').unique().sort()).collect().to_series().to_list()

    df_overlap_list = []

    for qry_id in qry_id_list:

        # Get overlaps

        df_overlap_list.append(
            df.filter(pl.col('qry_id') == qry_id)
            .join_where(
                df.filter(pl.col('qry_id') == qry_id),
                pl.col('_index') != pl.col('_index_right'),
                pl.col('qry_pos') < pl.col('qry_end_right'),
                pl.col('qry_end') > pl.col('qry_pos_right'),
                suffix='_right'
            )
        )

    df_overlap = pl.concat(df_overlap_list).collect()

    if df_overlap.height > 0:
        record_str = '; '.join(
            (
                'L=[align_index={align_index}, qry=({qry_id}, {qry_pos}, {qry_end})], '
                'R=[align_index={align_index_right}, qry=({qry_id_right}, {qry_pos_right}, {qry_end_right})]'
            ).format(**row)
            for row in df_overlap.head(3).rows(named=True)
        ) + '...' if df_overlap.height > 3 else ''

        raise ValueError(f'Found {df_overlap.height} overlapping query alignments: {record_str}')


def check_overlap_ref(
        df: pl.DataFrame
):
    """Check for alignment overlaps in reference coordinates and raise an exception if any are found.

    :param df: Alignment dataframe after query trimming.

    :raises ValueError: If overlaps in query coordinates are found.
    """
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    df = (
        df
        .drop('_index', strict=False)
        .with_row_index('_index')
    )

    chrom_list = df.select(pl.col('chrom').unique().sort()).collect().to_series().to_list()

    df_overlap_list = []

    for chrom in chrom_list:

        # Get overlaps

        df_overlap_list.append(
            df.filter(pl.col('chrom') == chrom)
            .join_where(
                df.filter(pl.col('chrom') == chrom),
                pl.col('_index') != pl.col('_index_right'),
                pl.col('pos') < pl.col('end_right'),
                pl.col('end') > pl.col('pos_right'),
                suffix='_right'
            )
        )

    df_overlap = pl.concat(df_overlap_list).collect()

    if df_overlap.height > 0:
        record_str = '; '.join(
            (
                'L=[align_index={align_index}, ref=({chrom}, {pos}, {end})], '
                'R=[align_index={align_index_right}, ref=({chrom_right}, {pos_right}, {end_right})]'
            ).format(**row)
            for row in df_overlap.head(3).rows(named=True)
        ) + '...' if df_overlap.height > 3 else ''

        raise ValueError(f'Found {df_overlap.height} overlapping reference alignments: {record_str}')
