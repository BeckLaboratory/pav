"""Variant anchor intervals.

An anchored interval spans between two anchors and covers segmentns of aligned and unaligned segments.
"""

__all__ = [
    'AnchoredInterval',
    'get_segment_table',
    'score_segment_transitions'
]

import polars as pl

from .. import schema

from ..align.features import FeatureGenerator
from ..region import Region

from .resources import CallerResources

SEGMENT_TABLE_SCHEMA = {
    'chrom': schema.ALIGN['chrom'],
    'pos': schema.ALIGN['pos'],
    'end': schema.ALIGN['end'],
    'filter': schema.ALIGN['filter'],
    'filter_pass': pl.Boolean,
    'is_anchor': pl.Boolean,
    'is_aligned': pl.Boolean,
    'qry_id': schema.ALIGN['qry_id'],
    'qry_pos': schema.ALIGN['qry_pos'],
    'qry_end': schema.ALIGN['qry_end'],
    'is_rev': schema.ALIGN['is_rev'],
    'len_ref': schema.ALIGN['pos'],
    'len_qry': schema.ALIGN['qry_pos'],
    'gap_ref': schema.ALIGN['pos'],
    'align_index': schema.ALIGN['align_index'],
    'qry_order': schema.ALIGN['qry_order'],
    'seg_index': pl.UInt32,
    'score': FeatureGenerator.all_feature_schema()['score'],
}
"""
Schema for segment tables.

The segment tables traces part of a query sequence through alignments. Segments are bounded by anchors, two confidently-
aligned segments anchoring a potential variant between them. Anchors are not part of a variant call, but they are
included in the segment table as the first and last segments.

All positions are 0-based, half-open intervals (BED-like coordinates).

Columns:
    * chrom: Reference chromosome (null for unaligned segments).
    * pos: Reference start position (null for unaligned segments).
    * end: Reference end position (null for unaligned segments).
    * filter: Alignment filters (null for unaligned segments).
    * filter_pass: True if the alignment passes all filters (if filter is empty, False for unaligned segments).
    * is_anchor: True if the alignment is an anchor.
    * is_aligned: True if the alignment is aligned.
    * qry_id: Assembly sequence ID.
    * qry_pos: Query start position.
    * qry_end: Query end position.
    * is_rev: True if the sequence was reverse-complemented during alignment.
    * len_ref: Length of the segment on the reference.
    * len_qry: Length of the segment on the query.
    * gap_ref: Gap length on the reference. Distance between the end of the last aligned segment and this segment. May
        be negative if the alignment went backwards. If the chromosome name of the last aligned segment is different,
        this value is null (undefined). Always 0 for unaligned segments.
    * align_index: Alignment index from the alignment table.
    * qry_order: Query order from the alignment table.
    * seg_index: Segment index. Starts at 0, increments for each segment including unaligned segments.
    * score: Alignment score from the alignment table.
"""


class AnchoredInterval:
    """Interval spanning from one anchoring alignment to another.

    Represents one interval of alignments spanning from one anchor to another. This object is fed into call methods.

    :ivar start_index: Interval start index
    :ivar end_index: Interval end index
    :ivar is_rev: True If both anchors are reverse-complemented and False if both are not (error if they don't match).
    :ivar df_segment: Table describing aligned and unaligned segments between two aligned anchors.
    :ivar region_ref: Reference region between the anchors.
    :ivar region_qry: Query region between the anchors.
    :ivar len_ref: Distance between the anchors on the reference. May be a negative number (i.e. Tandem Duplications).
    :ivar len_qry: Distance between the anchors on the query.
    """

    start_index: int
    end_index: int
    is_rev: bool
    df_segment: pl.DataFrame
    region_ref: Region
    region_qry: Region
    len_ref: int
    len_qry: int

    def __init__(
            self,
            start_index: int,
            end_index: int,
            df_align: pl.DataFrame,
            caller_resources: CallerResources
    ) -> None:
        """Initialize interval.

        :parm start_index: Interval start index.
        :parm end_index: Interval end index.
        :parm df_align: DataFrame of alignment records with the 'SCORE' column added.
        :parm caller_resources: Caller resources.
        """
        if end_index <= start_index:
            raise ValueError(f'End index {end_index} must be greater than the start index {start_index}')

        self.start_index = start_index
        self.end_index = end_index
        self.caller_resources = caller_resources

        head_row = df_align.row(start_index, named=True)
        tail_row = df_align.row(end_index, named=True)

        chrom = head_row['chrom']
        qry_id = head_row['qry_id']

        self.is_rev = head_row['is_rev']

        if (
                head_row['is_rev'] != tail_row['is_rev'] or
                head_row['qry_id'] != tail_row['qry_id'] or
                head_row['chrom'] != tail_row['chrom']
        ):
            raise ValueError(
                f'Anchors {start_index} (align_index={head_row["align_index"]}) and '
                f'{end_index} (align_index={tail_row["align_index"]}): '
                f'Non-matching is_rev, qry_id, or chrom'
            )

        # Get segment table
        self.df_segment = get_segment_table(start_index, end_index, df_align, caller_resources)

        # Get query and reference regions
        qry_pos = self.df_segment[0, 'qry_end']
        qry_end = self.df_segment[-1, 'qry_pos']

        if self.is_rev:
            ref_pos = self.df_segment[-1, 'end']
            ref_end = self.df_segment[0, 'pos']
        else:
            ref_pos = self.df_segment[0, 'end']
            ref_end = self.df_segment[-1, 'pos']

        # Get query region
        self.region_ref = Region(chrom, min(ref_pos, ref_end), max(ref_pos, ref_end))
        self.region_qry = Region(qry_id, qry_pos, qry_end, self.is_rev)

        self.len_ref = ref_end - ref_pos
        self.len_qry = len(self.region_qry)

        # Test invariants
        if self.len_qry < 0:
            raise ValueError(f'Qry region length is negative: {self.region_qry}')

        if self.seg_n != self.df_segment.shape[0] - 2:
            raise ValueError(
                f'Segment count mismatch: seg_n={self.seg_n} != segments - 2 ({self.df_segment.shape[0] - 2})'
            )

    @property
    def chrom(self) -> str:
        """Reference chromosome name."""
        return self.region_ref.chrom

    @property
    def qry_id(self) -> str:
        """Assembly sequenc ID."""
        return self.region_qry.chrom

    @property
    def qry_bp(self) -> int:
        """Number of query bases, aligned or unaligned, excluding anchors."""
        return (
            self.df_segment
            .filter(~ pl.col('is_anchor'))
            .select(pl.col('len_qry').sum())
            .item()
        )

    @property
    def qry_aligned_bp(self) -> int:
        """Number of aligned bases excluding anchors."""
        return (
            self.df_segment
            .filter(pl.col('is_aligned') & ~ pl.col('is_anchor'))
            .select(pl.col('len_qry').sum())
            .item()
        )

    @property
    def qry_aligned_bp_pass(self) -> int:
        """Number of aligned bases passing filters and excluding anchors."""
        return (
            self.df_segment
            .filter(pl.col('is_aligned') & ~ pl.col('is_anchor') & pl.col('filter_pass'))
            .select(pl.col('len_qry').sum())
            .item()
        )

    @property
    def qry_aligned_bp_fail(self) -> int:
        """Number of aligned bases failing filters and excluding anchors."""
        return (
            self.df_segment
            .filter(pl.col('is_aligned') & ~ pl.col('is_anchor') & ~ pl.col('filter_pass'))
            .select(pl.col('len_qry').sum())
            .item()
        )

    @property
    def qry_aligned_pass_prop(self):
        """Proportion of the aligned query bases in alignment records passing filters."""
        return (
            self.df_segment
            .filter(pl.col('is_aligned') & ~ pl.col('is_anchor'))
            .select(
                pl.when(pl.col('filter_pass')).then(pl.col('len_qry')).otherwise(0).sum() / pl.col('len_qry').sum()
            )
            .item()
        )

    @property
    def qry_aligned_prop(self) -> float:
        """Proportion of the query bases inside alignments (no anchors)."""
        return self.qry_aligned_bp / self.qry_bp if self.qry_bp > 0 else 0.0

    @property
    def qry_aligned_fail_prop(self) -> float:
        """Get the proportion of the aligned bases failing filters."""
        return self.qry_aligned_bp_fail / self.qry_aligned_bp if self.qry_aligned_bp > 0 else 0.0

    @property
    def seg_n(self) -> int:
        """Number of segments between the anchors."""
        return (
            self.df_segment
            .select((~ pl.col('is_anchor')).sum())
            .item()
        )

    @property
    def seg_n_aligned(self) -> int:
        """Number of segments between the anchors that are aligned."""
        return (
            self.df_segment
            .select((~ pl.col('is_anchor') & pl.col('is_aligned')).sum())
            .item()
        )

    @property
    def all_anchor_pass(self) -> bool:
        """True if all anchors pass filters."""
        return (
            self.df_segment
            .filter(pl.col('is_anchor'))
            .select(pl.col('filter_pass').all())
            .item()
        )

    @property
    def n_anchor(self) -> bool:
        """Number of anchors."""
        return (
            self.df_segment
            .select(pl.col('is_anchor').sum())
            .item()
        )

    @property
    def min_anchor_score(self) -> float:
        """Get the minimum score of both anchors."""
        return (
            self.df_segment
            .filter(pl.col('is_anchor'))
            .select(pl.col('score').min())
            .item()
        )

    @property
    def segment_transition_score(self) -> float:
        """Get a score for segment transitions."""
        return score_segment_transitions(self.df_segment, self.caller_resources)

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return (
            f'AnchoredInterval('
            f'start_index={self.start_index}, '
            f'end_index={self.end_index}, '
            f'ref={self.region_ref}, query={self.region_qry}, '
            f'seg_n={self.seg_n}, is_rev={self.is_rev})'
        )


def get_segment_table(
        start_index: int,
        end_index: int,
        df_align: pl.DataFrame,
        caller_resources: CallerResources
) -> pl.DataFrame:
    """Get a segment table from an interval.

    Generates a table of segments across between two alignment records, which may or may not be anchors for variant
    detection. In the case that the alignment records are not anchors, the table is not fully well-formed, but contains
    enough information to score a transition across alignment records.

    :param start_index: Index of the left-most aligned segment in query coordinate order.
    :param end_index: Index of the right-most aligned segment in query coordinate order.
    :param df_align: Table of trimmed alignments.
    :param caller_resources: Caller resources.

    :returns: A segment table with aligned and unaligned segments from `start_index` to `end_index` (inclusive).
    """
    score_model = caller_resources.score_model

    # Create templated and untemplated insertions for each aligned segment and transition.
    # Save row_l outside the loop, row_r may be transformed inside the loop and should be preserved for the next round
    # as row_r.
    index_l = start_index
    row_l = df_align.row(index_l, named=True)

    index_r = index_l + 1

    segment_list = []  # List of query segments broken by template switches and untemplated insertions

    head_row = df_align.row(start_index, named=True)
    tail_row = df_align.row(end_index, named=True)

    is_rev = head_row['is_rev']
    qry_id = head_row['qry_id']

    # Add head anchor
    segment_list.append(
        {
            col: head_row[col] for col in [
                'chrom', 'pos', 'end', 'qry_id', 'qry_pos', 'qry_end',
                'filter', 'is_rev',  'align_index', 'qry_order', 'score'
            ]
        } | {
            'is_anchor': True,
            'is_aligned': True,
            'len_ref': head_row['end'] - head_row['pos'],
            'len_qry': head_row['qry_end'] - head_row['qry_pos'],
            'gap_ref': 0
        }
    )

    # Traverse alignment records setting segments
    while index_r <= end_index:

        # Get right row
        row_r = df_align.row(index_r, named=True)

        if row_r['qry_id'] != qry_id:
            raise ValueError(
                f'Query ID in alignment record {row_r["align_index"]} ("{row_r["qry_id"]}") '
                f'does not match the query ID for this interval "{qry_id}"'
            )

        # Get query and reference positions
        gap_qry_pos = row_l['qry_end']
        gap_qry_end = row_r['qry_pos']

        if is_rev:  # Get rows in reference order
            row_l_ref = row_r
            row_r_ref = row_l
        else:
            row_l_ref = row_l
            row_r_ref = row_r

        gap_ref_pos = row_l_ref['end']
        gap_ref_end = row_r_ref['pos']

        gap_len_qry = gap_qry_end - gap_qry_pos
        gap_len_ref = gap_ref_end - gap_ref_pos if (row_l_ref['chrom'] == row_r_ref['chrom']) else None

        if gap_len_qry < 0:
            raise ValueError(f'Negative query gap between alignment records {index_l} and {index_r}: {gap_len_qry}')

        if gap_len_qry > 0:
            # Add gap between template switches as an untemplated insertion
            segment_list.append(
                {
                    'chrom': None, 'pos': None, 'end': None,
                    'filter': None,
                    'is_anchor': False,
                    'is_aligned': False,
                    'qry_id': qry_id, 'qry_pos': gap_qry_pos, 'qry_end': gap_qry_end,
                    'is_rev': None,
                    'len_ref': 0,
                    'len_qry': gap_len_qry,
                    'gap_ref': 0,
                    'align_index': None,
                    'qry_order': None,
                    'score': score_model.gap(gap_len_qry),
                }
            )

        # Add templated segment
        if index_r != end_index:
            segment_list.append(
                {
                    col: row_r[col] for col in [
                        'chrom', 'pos', 'end', 'qry_id', 'qry_pos', 'qry_end',
                        'filter', 'is_rev',  'align_index', 'qry_order', 'score'
                    ]
                } | {
                    'is_anchor': False,
                    'is_aligned': True,
                    'len_ref': row_r['end'] - row_r['pos'],
                    'len_qry': row_r['qry_end'] - row_r['qry_pos'],
                    'gap_ref': gap_len_ref
                }
            )

        # Next aligned segment
        index_l = index_r
        index_r = index_l + 1

        row_l = row_r

    # Add tail anchor
    segment_list.append(
        {
            col: tail_row[col] for col in [
                'chrom', 'pos', 'end', 'qry_id', 'qry_pos', 'qry_end',
                'filter', 'is_rev',  'align_index', 'qry_order', 'score'
            ]
        } | {
            'is_anchor': True,
            'is_aligned': True,
            'len_ref': tail_row['end'] - tail_row['pos'],
            'len_qry': tail_row['qry_end'] - tail_row['qry_pos'],
            'gap_ref': 0
        }
    )

    # Concat segments, order by query.
    # Set complex reference position start/end by the anchoring alignment records.
    return (
        pl.from_dicts(segment_list, schema_overrides=SEGMENT_TABLE_SCHEMA)
        .lazy()
        .with_row_index('seg_index')
        .with_columns(
            (pl.col('filter').list.len() == 0).alias('filter_pass').cast(SEGMENT_TABLE_SCHEMA['filter_pass']),
        )
        .cast(SEGMENT_TABLE_SCHEMA)
        .select(
            list(SEGMENT_TABLE_SCHEMA.keys())
        )
        .collect()
    )


def score_segment_transitions(
        df_segment: pl.DataFrame,
        caller_resources: CallerResources,
) -> float:
    """Scores a complex variant on template switches and gaps for each segment.

    :param df_segment: Segment table.
    :param caller_resources: Caller resources.

    :returns: Variant score.
    """
    # Template switches between segments (n + 2 alignment records (n segments + 2 anchors),
    # n - 1 template switches (between each segment including each anchor)
    score_variant = (
            caller_resources.score_model.template_switch() *
            (df_segment.height - 1)
    )

    return score_variant + (
        df_segment
        .filter(pl.col('is_aligned') & ~ pl.col('is_anchor'))
        .select(pl.col('len_qry').map_elements(caller_resources.score_model.gap, return_dtype=pl.Float32).sum())
        .item()
    )
