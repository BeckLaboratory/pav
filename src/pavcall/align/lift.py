"""Alignmnet lift utility.

Alignment lift operations translate between reference and query alignments (both directions) using alignment tables.
"""

__all__ = [
    'AlignLift',
]

from dataclasses import dataclass, field
import collections
import numbers

import intervaltree
import polars as pl
from typing import Iterable, Any

from .. import schema

from ..region import Region
from ..seq import seq_len

from . import op

from .features import FeatureGenerator


@dataclass(frozen=True, repr=False)
class AlignLift:
    """Alignment liftover utility.

    Create an alignment liftover object for translating between reference and query alignments (both directions). Build
    liftover from alignment data in a DataFrame (requires chrom, pos, end, qry_id, qry_pos, and qry_end).

    :ivar df: Alignment DataFrame.
    :ivar df_qry_fai: Query FAI DataFrame.
    :ivar cache_align: Number of alignment records to cache in memory.
    """

    df: pl.DataFrame
    df_qry_fai: pl.DataFrame
    cache_align: int = 10

    # Maps reference loci to an alignment index
    _ref_tree: intervaltree.IntervalTree = field(
        default_factory=lambda: collections.defaultdict(intervaltree.IntervalTree), init=False, repr=False
    )

    # Maps query loci to an alignment index
    _qry_tree: intervaltree.IntervalTree = field(
        default_factory=lambda: collections.defaultdict(intervaltree.IntervalTree), init=False, repr=False
    )

    _cache_queue: collections.deque = field(
        default_factory=collections.deque, init=False, repr=False
    )

    _ref_cache: dict = field(
        default_factory=dict, init=False, repr=False
    )

    _qry_cache: dict = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        """Complete initialization."""
        # Add row index
        expected_cols = {'chrom', 'pos', 'end', 'qry_id', 'qry_pos', 'qry_end', 'is_rev'}
        auto_cols = {'score', 'align_index'}

        if missing_cols := expected_cols - set(self.df.collect_schema().names()):
            raise ValueError(f'Missing columns: {", ".join(sorted(missing_cols))}')

        add_cols = auto_cols - set(self.df.collect_schema().names())

        if add_cols:
            df = self.df.lazy()

            if 'align_index' in add_cols:
                df.with_columns(pl.int_range(0, self.df.height).alias('align_index').cast(schema.ALIGN['align_index']))

            if 'score' in add_cols:
                df = df.with_columns(
                    pl.lit(0.0).cast(
                        FeatureGenerator.all_feature_schema()['score']
                    )
                )

            object.__setattr__(self, 'df', (
                df.collect()
            ))

        # Load trees
        for row in self.df.rows(named=True):
            self._ref_tree[row['chrom']][row['pos']:row['end']] = row['align_index']
            self._qry_tree[row['qry_id']][row['qry_pos']:row['qry_end']] = row['align_index']

    def coord_to_ref(
            self,
            query_id: str,
            coord: int | Iterable[int],
    ):
        """Lift coordinates from query to reference.

        For each coordinate, multiple lift locations may be found, so a list of dicts is returned for each `coord`
        value.

        Each dict has the following keys:
        - chrom: Query ID
        - pos: Query position
        - is_rev: Whether the query is reverse-complemented
        - align_index: Alignment align_index
        - score: Alignment score

        If `coord` is a single value, then a single list of dicts is returned. If `coord` is iterable, then a list of
        lists is returned (one for each element in `coord`). If lift fails for a coordinate, then an empty list is
        returned for that element.

        :param query_id: Query record ID.
        :param coord: Query coordinates. May be a single value or an iterable.

        :returns: A list of dicts or a list of lists of dicts, depending on the type of `coord` (see above).
        """
        # Determine type
        if isinstance(coord, numbers.Integral):
            ret_list = False
            coord = (coord,)
        else:
            coord = tuple(coord)
            ret_list = True

        # Do lift
        lift_coord_list = []

        for pos in coord:
            pos_org = pos  # Pre-reverse position for error reporting

            match_list = []

            for match_element in self._qry_tree[query_id][pos:(pos + 1)]:
                pos = pos_org

                # Get lift tree
                align_index = match_element.data

                if align_index not in self._qry_cache.keys():
                    self._add_align(align_index)

                lift_tree = self._qry_cache[align_index]

                # Get row
                row = self.df.row(by_predicate=pl.col('align_index') == align_index, named=True)

                # Reverse coordinates of pos if the alignment is reverse-complemented.
                if row['is_rev']:
                    pos = seq_len(query_id, self.df_qry_fai) - pos

                # Get match record
                lift_set = lift_tree[pos:(pos + 1)]

                if len(lift_set) != 1:
                    raise ValueError(
                        f'Program bug: Expected one match in a lift-tree for a record withing a '
                        f'global to-ref tree: {query_id}:{pos_org} (align_index={align_index}): Found {len(lift_set)}'
                    )

                lift_interval = list(lift_set)[0]

                # Interpolate coordinates
                assert pos >= lift_interval.begin, 'Expected pos >= lift_interval.begin: %d >= %d' % (
                    pos, lift_interval.begin
                )

                if lift_interval.data[1] - lift_interval.data[0] > 1:
                    ref_pos = lift_interval.data[0] + (pos - lift_interval.begin)
                else:
                    ref_pos = lift_interval.data[1]

                match_list.append({
                    'chrom': row['chrom'],
                    'pos': ref_pos,
                    'is_rev': row['is_rev'],
                    'align_index': row['align_index'],
                    'score': row['score']
                })

            # Append
            lift_coord_list.append(match_list)

        # Return coordinates
        if ret_list:
            return lift_coord_list
        else:
            return lift_coord_list[0]

    def coord_to_qry(
            self,
            chrom: str,
            coord: int | Iterable[int]
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
        """Lift coordinates from reference to query.

        For each coordinate, multiple lift locations may be found, so a list of dicts is returned for each `coord`
        value.

        Each dict has the following keys:
        - chrom: Query ID
        - pos: Query position
        - is_rev: Whether the query is reverse-complemented
        - align_index: Alignment align_index
        - score: Alignment score

        If `coord` is a single value, then a single list of dicts is returned. If `coord` is iterable, then a list of
        lists is returned (one for each element in `coord`). If lift fails for a coordinate, then an empty list is
        returned for that element.

        :param chrom: Reference chromosome ID.
        :param coord: Query coordinates. May be a single value or an iterable.

        :returns: A list of dicts or a list of lists of dicts, depending on the type of `coord` (see above).
        """
        # Determine type
        if isinstance(coord, numbers.Integral):
            coord = (coord, )
            ret_list = False
        else:
            coord = tuple(coord)
            ret_list = True

        # Do lift
        lift_coord_list = []

        for pos in coord:
            match_list = []

            for match_element in self._ref_tree[chrom][pos:(pos + 1)]:

                # Get lift tree
                align_index = match_element.data

                if align_index not in self._ref_cache.keys():
                    self._add_align(align_index)

                lift_tree = self._ref_cache[align_index]

                # Get row
                row = self.df.row(by_predicate=pl.col('align_index') == align_index, named=True)

                # Get match record
                lift_set = lift_tree[pos:(pos + 1)]

                if len(lift_set) != 1:
                    raise ValueError(
                        f'Program bug: Expected one match in a lift-tree for a record withing a '
                        f'global to-query tree: {chrom}:{pos} (align_index={align_index}): Found {len(lift_set)}'
                    )

                lift_interval = list(lift_set)[0]

                # Interpolate coordinates
                assert pos >= lift_interval.begin, 'Expected pos >= lift_interval.begin: %d >= %d' % (
                    pos, lift_interval.begin
                )

                if lift_interval.data[1] - lift_interval.data[0] > 1:
                    qry_pos = lift_interval.data[0] + (pos - lift_interval.begin)

                else:  # Lift from missing bases on the target (insertion or deletion)
                    qry_pos = lift_interval.data[1]

                if row['is_rev']:
                    qry_pos = seq_len(row['qry_id'], self.df_qry_fai) - qry_pos

                match_list.append({
                    'chrom': row['qry_id'],
                    'pos': qry_pos,
                    'is_rev': row['is_rev'],
                    'align_index': row['align_index'],
                    'score': row['score']
                })

            # Add records for this position
            lift_coord_list.append(match_list)

        # Return coordinates
        if ret_list:
            return lift_coord_list
        else:
            return lift_coord_list[0]

    def region_to_ref(
            self,
            region_qry: Region,
            same_index: bool = False
    ):
        """Lift region to reference.

        :param region_qry: Query region.
        :param same_index: If True, the region must have the same alignment index as the reference region.

        :returns: Reference region or `None` if the query region could not be lifted.
        """
        # Lift
        ref_pos, ref_end = self.coord_to_ref(region_qry.chrom, (region_qry.pos, region_qry.end))

        if same_index:
            if region_qry.pos_align_index is not None:
                ref_pos = [_ for _ in ref_pos if _['align_index'] == region_qry.pos_align_index]

            if region_qry.end_align_index is not None:
                ref_end = [_ for _ in ref_end if _['align_index'] == region_qry.end_align_index]

        if len(ref_pos) != 1 or len(ref_end) != 1:
            return None

        ref_pos = ref_pos[0]
        ref_end = ref_end[0]

        if ref_pos['chrom'] != ref_end['chrom'] or ref_pos['is_rev'] != ref_end['is_rev']:
            return None

        # Return
        return Region(
            chrom=ref_pos['chrom'],
            pos=min(ref_pos['pos'], ref_pos['pos']),
            end=max(ref_pos['pos'], ref_pos['pos']),
            is_rev=ref_pos['is_rev'],
            pos_align_index=ref_pos['align_index'],
            end_align_index=ref_end['align_index'],
        )

    def region_to_qry(
            self,
            region_ref: Region,
            same_index: bool = False
    ):
        """Lift region to query.

        :param region_ref: Reference region.
        :param same_index: If True, the region must have the same alignment index as the reference region.

        :returns: Query region or `None` if it could not be lifted.
        """
        # Lift
        query_pos, query_end = self.coord_to_qry(region_ref.chrom, (region_ref.pos, region_ref.end))

        if same_index:
            if region_ref.pos_align_index is not None:
                query_pos = [_ for _ in query_pos if _['align_index'] == region_ref.pos_align_index]

            if region_ref.end_align_index is not None:
                query_end = [_ for _ in query_end if _['align_index'] == region_ref.end_align_index]

        if len(query_pos) != 1 or len(query_end) != 1:
            return None

        query_pos = query_pos[0]
        query_end = query_end[0]

        if query_pos['chrom'] != query_end['chrom'] or query_pos['is_rev'] != query_end['is_rev']:
            return None

        # Return
        return Region(
            chrom=query_pos['chrom'],
            pos=min(query_pos['pos'], query_end['pos']),
            end=max(query_pos['pos'], query_end['pos']),
            is_rev=query_pos['is_rev'],
            pos_align_index=query_pos['align_index'],
            end_align_index=query_end['align_index'],
        )

    def _add_align(self, index):
        """Add an alignment from DataFrame index `index`.

        :param index: DataFrame index.

        :raises ValueError: If an unhandled alignment operation is found.
        """
        # No alignment to add if it's already cached.
        if index in self._ref_cache.keys():

            # Append to end (last used) and skip adding alignment
            while index in self._cache_queue:
                self._cache_queue.remove(index)

            self._cache_queue.appendleft(index)

            return

        # Make space for this alignment
        self._check_and_clear()

        # Get row
        row = self.df.row(by_predicate=pl.col('align_index') == index, named=True)

        # Build lift trees
        ref_pos = row['pos']
        qry_pos = 0

        itree_ref = intervaltree.IntervalTree()
        itree_qry = intervaltree.IntervalTree()

        op_arr = op.row_to_arr(row)

        # Build trees
        for op_code, op_len in op_arr:
            op_len = int(op_len)

            if op_code in op.ALIGN_SET:
                ref_end = ref_pos + op_len
                qry_end = qry_pos + op_len

                itree_ref[ref_pos:ref_end] = (qry_pos, qry_end)
                itree_qry[qry_pos:qry_end] = (ref_pos, ref_end)

                ref_pos = ref_end
                qry_pos = qry_end

            elif op_code == op.I:
                qry_end = qry_pos + op_len

                itree_qry[qry_pos:qry_end] = (ref_pos, ref_pos + 1)

                qry_pos = qry_end

            elif op_code == op.D:
                ref_end = ref_pos + op_len

                itree_ref[ref_pos:ref_end] = (qry_pos, qry_pos + 1)

                ref_pos = ref_end

            elif op_code in op.CLIP_SET:
                qry_pos += op_len

            else:
                raise ValueError(
                    'Unhandled operation: {op_code}: Alignment {chrom}:{pos}-{end} (qry_id{qry_id})'.format(
                        op_code=op_code, **row
                    )
                )

        # Cache trees
        self._ref_cache[index] = itree_ref
        self._qry_cache[index] = itree_qry

        # Add index to end of queue
        self._cache_queue.appendleft(index)

    def _check_and_clear(self):
        """Check alignment cache and clear if necessary to make space."""
        while len(self._cache_queue) >= self.cache_align:
            index = self._cache_queue.pop()

            del self._ref_cache[index]
            del self._qry_cache[index]

    def __repr__(self):
        """Get a string representation."""
        return 'AlignLift()'
