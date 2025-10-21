"""Chain through alignment records.

Generate chains of potential variants. Each chain is an interval from one anchoring alignmnet to another representing
a potential structural rearrangement.
"""

__all__ = [
    'get_chain_set',
    'can_anchor',
    'can_reach_anchor',
    'get_min_anchor_score'
]

import polars as pl
from typing import Any

from ..align.score import ScoreModel
from ..const import DEFAULT_LG_GAP_SCALE

from .resources import CallerResources


def get_chain_set(
        df_align: pl.DataFrame,
        caller_resources: CallerResources,
        min_anchor_score: float = None,
) -> set[tuple[int, int]]:
    """Identify anchors.

    Find alignments that may "anchor" variants. Return a set of intervals where each edge of the interval
    is an anchor.

    :param df_align: Query alignment records for one query sequence sorted and indexed in query order.
    :param caller_resources: Caller resources.
    :param min_anchor_score: Minimum alignment score for anchors.

    :returns: A set of anchor intervals.
    """
    chain_set = set()

    start_index = 0
    last_index = df_align.height

    # Traverse interval setting each position to the left-most anchor candidate.
    while start_index < last_index:

        # Skip if anchor did not pass TIG & REF trimming
        if not df_align[start_index, 'in_qryref']:
            start_index += 1
            continue

        start_row = df_align.row(start_index, named=True)
        end_index = start_index + 1

        # Traverse each interval after the start for right-most anchor candidates. Limit search by contig distance
        while (
                end_index < last_index and
                can_reach_anchor(start_row, df_align.row(end_index, named=True), caller_resources.score_model)
        ):
            end_row = df_align.row(end_index, named=True)

            if end_row['in_qryref'] and can_anchor(
                    start_row, end_row, caller_resources.score_model, min_anchor_score,
                    gap_scale=caller_resources.pav_params.lg_gap_scale
            ):
                chain_set.add((start_index, end_index))

            end_index += 1

        start_index += 1

    return chain_set


def can_anchor(
        row_a: dict[str, Any],
        row_b: dict[str, Any],
        score_model: ScoreModel,
        min_score: float = 100.0,
        gap_scale: float = DEFAULT_LG_GAP_SCALE
) -> bool:
    """Determine if two alignment rows can anchor a rearrangement.

    Requires the "SCORE" column is added to the alignment
    rows.

    Both rows are alignment records representing a candidate poir of alignments for anchoring an SV
    (simple or complex) between them. If either row is `None`, they are not aligned to the same reference sequence, or
    if they are not aligned in the same orientation, `False` is returned.

    Each row has an alignment score ("score" column). If either anchor's score is less than `min_score`, `False`
    is returned. The minimum value of the pair alignment scores (from row_a and row_b) is compared to the gap between
    them. The gap between the alignments is found by adding the number of query bases skipped between the alignment
    records and the number of reference bases skipped moving from `row_a` to `row_b` (orientation does not matter, may
    skip forward (DEL) or backward (DUP)).

    The gap is scored as one gap with `score_model`. If the gap penalty exceeds the minimum alignment score of the
    anchors, then `False` is returned. Otherwise, `True` is returned and these alignments can function as anchors for
    an SV between them.

    :param row_a: Row earlier in the alignment chain (query position order).
    :param row_b: Row later in the alignment chain (query position order).
    :param score_model: Alignment scoring model used to score the gap between two rows (ref gap and qry gap).
    :param min_score: Cannot be an anchor if either anchor's alignment score is less than this value.
    :param gap_scale: Scale gap score by this factor. A value of less than 1 reduces the gap penalty (e.g. 0.5 halves
        it), and a value greater than 1 increases the gap penalty (e.g. 2.0 doubles it).

    :returns: `True` if rows are collinear in query and reference space.

    :raises ValueError: If alignment rows are not in query order.
    """
    # Both rows should be present and in the same orientation
    if row_a is None or row_b is None or row_a['chrom'] != row_b['chrom']:
        return False

    is_rev = row_a['is_rev']

    if is_rev != row_b['is_rev']:
        return False

    anchor_score = min([row_a['score'], row_b['score']])

    if anchor_score < min_score:
        return False

    # Check reference contiguity
    if is_rev:
        ref_l_end = row_b['end']
        ref_r_pos = row_a['pos']
    else:
        ref_l_end = row_a['end']
        ref_r_pos = row_b['pos']

    # Score gap
    gap_len = row_b['qry_pos'] - row_a['qry_end']

    if gap_len < 0:
        raise ValueError(
            f'Alignment rows are out of order: '
            f'Negative distance {gap_len}: row_a index "{row_a["align_index"]}", row_b index "{row_b["align_index"]}"'
        )

    gap_len += abs(ref_r_pos - ref_l_end)

    if gap_len == 0:
        return True

    return score_model.gap(gap_len) * gap_scale + anchor_score > 0


def can_reach_anchor(
        row_l: dict[str, Any],
        row_r: dict[str, Any],
        score_model: ScoreModel,
) -> bool:
    """Determine if a variant could span an interval.

    Determine if a left-most anchor can reach as far as a right-most anchor. This function only tells the traversal
    algorithm when to stop searching for further right-most anchor candidates, it does not determine if the two
    alignments are anchors (does not consider the score of the right-most alignment or the reference position or
    orientation).

    :param row_l: Left-most anchor in query coordinates.
    :param row_r: Right-most anchor in query coordinates.
    :param score_model: Model for determining a gap score.

    :returns: `True` if the anchor in alignment `row_l` can reach as far as the start of `row_r` based on the alignment
        score of `row_l` and the distance in query coordinates between the end of `row_l` and the start of `row_r`.
    """
    # Check rows
    if row_l is None or row_r is None:
        raise ValueError('Cannot score query distance for records "None"')

    if row_l['qry_id'] != row_r['qry_id']:
        raise ValueError(
            f'Cannot score query distance for mismatching queries: "{row_l["qry_id"]}" and "{row_r["qry_id"]}"'
        )

    qry_dist = row_r['qry_pos'] - row_l['qry_end']

    if qry_dist < 0:
        raise ValueError(
            f'Cannot score query distance for out-of-order (by query coordinates): '
            f'row_a index "{row_l["align_index"]}", row_b index "{row_r["align_index"]}"'
        )

    # Cannot anchor if query distance is too large
    return qry_dist == 0 or row_l['score'] + score_model.gap(qry_dist) > 0


def get_min_anchor_score(
        min_anchor_score: str | int | float,
        score_model: ScoreModel
) -> float:
    """Get the minimum score of an anchoring alignment.

    The score may be expressed as an absolute alignment score (value is numeric or a string representing a number), or a
    number of matching basepairs (string ending in "bp").

    Each large variant is anchored by a pair of alignment records where the large variant appears between them.
    Anchoring alignments must be sufficiently confident or evidence for the variant is not well-supported.

    :param min_anchor_score: Minimum score of an anchoring alignment. The score may be expressed as an absolute
        alignment score (numeric or string representing a number), or a number of matching basepairs
        (string ending in "bp").
    :param score_model: Score model to use for converting basepair scores to alignment scores.

    :returns: Minimum score for anchoring alignments.

    :raises ValueError: If min_anchor_score is not a string or numeric.
    """
    if isinstance(min_anchor_score, str):
        min_anchor_score_str = min_anchor_score.strip()

        if min_anchor_score_str.lower().endswith('bp'):
            try:
                min_anchor_score_bp = int(min_anchor_score_str[:-2].strip())
            except ValueError:
                raise ValueError(f'min_anchor_score: "bp" must come before an integer: "{min_anchor_score}"')

            if min_anchor_score_bp < 0:
                raise ValueError(
                    f'min_anchor_score: "bp" must come before a non-negative integer: "{min_anchor_score}"'
                )

            if score_model is None:
                raise ValueError('score_model is None')

            return float(score_model.match(min_anchor_score_bp))

        else:
            try:
                return abs(float(min_anchor_score))
            except ValueError:
                raise ValueError(
                    f'min_anchor_score is a string that does not represent a numeric value: {min_anchor_score}'
                )

    else:
        try:
            # noinspection PyTypeChecker
            return float(min_anchor_score)
        except ValueError:
            raise ValueError(f'min_anchor_score is not a string or numeric: type={type(min_anchor_score)}')
