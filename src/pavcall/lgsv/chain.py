"""
Alignment chaining functions
"""

from .. import const as pav_const

from . import util as pavcall_lgsv_util


class AnchorChainNode:
    """
    Describes an SV (simple or complex) anchored by aligned query bases on each side (start_index and end_index).

    :param start_index: Index of the left-most aligned segment in query coordinate order.
    :param start_index: Index of the right-most aligned segment in query coordinate order.
    """

    def __init__(self, start_index, end_index):
        self.start_index = start_index
        self.end_index = end_index

        self.next_node_list = list()

    def __repr__(self):
        """
        :return: String representation of a choin object.
        """

        return f'AnchorChainNode(start={self.start_index}, end={self.end_index}, children={self.n_children()})'

    def n_children(self):
        """
        Get the number of child nodes from this node.

        :return: Number if child nodes.
        """
        return len(self.next_node_list)

def get_chain_set(df_align, caller_resources, min_anchor_score=None):
    """
    Identify anchors (alignments that may "anchor" variants). Return a set of intervals where each edge of the interval
    is an anchor.

    :param df_align: Query alignment records for one query sequence sorted and indexed in query order.
    :param caller_resources: Caller resources.
    :param min_anchor_score: Minimum alignment score for anchors.

    :return: A set of anchor intervals.
    """

    chain_set = set()

    start_index = 0
    last_index = df_align.shape[0]

    qryref_index_set = set(caller_resources.df_align_qryref.index)

    if min_anchor_score is None:
        min_anchor_score = pavcall_lgsv_util.get_min_anchor_score(min_anchor_score, caller_resources.score_model)

    # Traverse interval setting each position to the left-most anchor candidate.
    while start_index < last_index:

        # Skip if anchor did not pass TIG & REF trimming
        if df_align.loc[start_index]['INDEX'] not in qryref_index_set:
            start_index += 1
            continue

        start_row = df_align.loc[start_index]
        end_index = start_index + 1

        # Traverse each interval after the start for right-most anchor candidates. Limit search by contig distance
        while end_index < last_index and can_reach_anchor(start_row, df_align.loc[end_index], caller_resources.score_model):

            if df_align.loc[end_index]['INDEX'] in qryref_index_set and can_anchor(
                    start_row, df_align.loc[end_index], caller_resources.score_model, min_anchor_score,
                    gap_scale=caller_resources.pav_params.lg_gap_scale
            ):
                chain_set.add((start_index, end_index))

            end_index += 1

        start_index += 1

    return chain_set


def can_anchor(row_a, row_b, score_model, min_score=100, gap_scale=pav_const.DEFAULT_LG_GAP_SCALE):
    """
    Determine if two alignment rows can anchor a rearrangement. Requires the "SCORE" column is added to the alignment
    rows.

    Both rows are records from an alignment BED file representing a candidate poir of alignments for anchoring an SV
    (simple or complex) between them. If either row is `None`, they are not aligned to the same reference sequence, or
    if they are not aligned in the same orientation, `False` is returned.

    Each row has an alignment score ("SCORE" column), which is computed if the "SCORE" column is absent using
    `score_model`. If either anchor's score is less than `min_score`, `False` is returned. The minimum value of the pair
    alignment scores (from row_a and row_b) is compared to the gap between them. The gap between the alignments is found
    by adding the number of query bases skipped between the alignment records and the number of reference bases skipped
    moving from `row_a` to `row_b` (orientation does not matter, may skip forward (DEL) or backward (DUP)).

    The gap is scored as one gap with `score_model`. If the gap penalty exceeds the minimum alignment score of the
    anchors, then `False` is returned. Otherwise, `True` is returned and these alignments can function as anchors for
    an SV between them.

    :param row_a: Row earlier in the alignment chain (query position order).
    :param row_b: Row later in the alignment chain (query position order).
    :param score_model: Alignment scoring model used to score the gap between two rows (ref gap and qry gap).
    :param min_score: Cannot be an anchor if either anchor's alignment score is less than this value.
    :param gap_scale: Scale gap score by this factor. A value of less than 1 reduces the gap penalty (e.g. 0.5 halves
        it), and a value greater than 1 increases the gap penalty (e.g. 2.0 doubles it).

    :return: `True` if both rows are not `None` and are collinear in query and reference space.
    """

    # Both rows should be present and in the same orientation
    if row_a is None or row_b is None or row_a['#CHROM'] != row_b['#CHROM']:
        return False

    is_rev = row_a['IS_REV']

    if is_rev != row_b['IS_REV']:
        return False

    anchor_score = min([row_a['SCORE'], row_b['SCORE']])

    if anchor_score < min_score:
        return False

    # Check reference contiguity
    if is_rev:
        ref_l_end = row_b['END']
        ref_r_pos = row_a['POS']
    else:
        ref_l_end = row_a['END']
        ref_r_pos = row_b['POS']

    # Score gap
    gap_len = row_b['QRY_POS'] - row_a['QRY_END']

    if gap_len < 0:
        raise RuntimeError(f'Alignment rows are out of order: Negative distance {gap_len}: row_a index "{row_a.name}", row_b index "{row_b.name}"')

    gap_len += abs(ref_r_pos - ref_l_end)

    if gap_len == 0:
        return True

    return score_model.gap(gap_len) * gap_scale + anchor_score > 0


def can_reach_anchor(row_l, row_r, score_model):
    """
    Determine if a left-most anchor can reach as far as a right-most anchor. This function only tells the traversal
    algorithm when to stop searching for further right-most anchor candidates, it does not determine if the two
    alignments are anchors (does not consider the score of the right-most alignment or the reference position or
    orientation).

    :param row_l: Left-most anchor in query coordinates.
    :param row_r: Right-most anchor in query coordinates.
    :param score_model: Model for determining a gap score.

    :return: `True` if the anchor in alignment `row_l` can reach as far as the start of `row_r` based on the alignment
        score of `row_l` and the distance in query coordinates between the end of `row_l` and the start of `row_r`.
    """

    # Check rows
    if row_l is None or row_r is None:
        raise RuntimeError('Cannot score query distance for records "None"')

    if row_l['QRY_ID'] != row_r['QRY_ID']:
        raise RuntimeError(f'Cannot score query distance for mismatching queries: "{row_l["QRY_ID"]}" and "{row_r["QRY_ID"]}"')

    qry_dist = row_r['QRY_POS'] - row_l['QRY_END']

    if qry_dist < 0:
        raise RuntimeError(f'Cannot score query distance for out-of-order (by query coordinates): indexes "{row_l["INDEX"]}" and "{row_r["INDEX"]}"')

    # Cannot anchor if query distance is too large
    return qry_dist == 0 or row_l['SCORE'] + score_model.gap(qry_dist) > 0
