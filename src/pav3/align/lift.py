"""Alignmnet lift utility.

Alignment lift operations translate between reference and query alignments (both directions) using alignment tables.
"""

__all__ = [
    'AlignLift',
    'DEFAULT_CACHE_SIZE',
]

from functools import lru_cache

from agglovar.meta.decorators import immutable
import polars as pl
import polars.selectors as cs
from typing import (
    Any,
    Iterable,
    NamedTuple,
    Optional,
)

from .. import schema

from ..region import Region
from ..seq import seq_len

from . import op

from .features import FeatureGenerator

DEFAULT_CACHE_SIZE: int = 100
"""Default number of liftover tree pairs (one per alignment record) to cache in memory."""


class LiftRange(NamedTuple):
    """Elements of a liftover tree.

    :ivar pos: Query position.
    :ivar end: Query end position.
    :ivar op_code: Query position code.
    """
    pos: int
    end: int
    op_code: int

    @property
    def op_char(self) -> Optional[str]:
        """Return the operation character.

        :returns: Operation character or None if the operation is unknown.
        """
        return op.OP_CHAR.get(self.qp_code, None)

    @property
    def is_aligned(self) -> bool:
        """Return True if the operation is a match."""
        return self.op_code in op.ALIGN_SET

    @property
    def is_eq(self) -> bool:
        """Return True if the operation is a match."""
        return self.op_code == op.EQ

    @property
    def is_x(self) -> bool:
        """Return True if tthe operation is a mismatch."""
        return self.op_code == op.X


@immutable
class LiftSeg:
    """Alignment lift segment describing one alignment record.

    Attributes include `align_table_index`, which is the index of the table where the alignment
    row was (first row is 0 and increments by 1 for each row); this differs from the "align_index"
    column in the alignment table.

    Additional attributes are "row" for a copy of the alignment row and any column name in the row
    is an attribute to the object (e.g. "chrom" and "is_rev" are extracted from the row).

    :ivar align_table_index: Alignment table index where the first row is 0, etc. Not the same as "align_index" in
        the alignment table.
    :ivar qry_len: Full length of the full query sequence aligned in this record (not just the aligned segment).
    :ivar is_rev: True if the alignment is reverse complemented.
    :ivar df_ops: Expanded alignment operations with reference and query coordinates.
    """
    align_table_index: int
    """Index of the alignment record in the alignment table (first row is 0, etc.)."""

    qry_len: int
    """Full length of the full query sequence aligned in this record (not just the aligned segment)."""

    def __init__(
            self,
            align_table_index: int,
            df: pl.LazyFrame,
            df_index: pl.DataFrame,
            fai_dict: dict[str, int],
    ):
        # Check and set base parameters
        self.align_table_index = align_table_index
        self._row = df_index.row(align_table_index, named=True)

        if self._row is None:
            raise ValueError(f'Alignment record at row {align_table_index} not found')

        try:
            self.qry_len = fai_dict[self._row['qry_id']]
        except KeyError:
            raise ValueError(f'Query sequence {self._row["qry_id"]} not found in query FAI')

        self.df_ops, self.is_rev = _get_df_op(
            align_table_index,
            df,
            self.qry_len,
        )

        assert self.is_rev == self._row['is_rev'], 'is_rev mismatch among tables'

        assert (
                   qry_pos_min := self.df_ops.select(pl.col('qry_pos').min()).item()
               ) == self._row['qry_pos'], (
            f'Query start position mismatch in LiftSeg init: lift-table={qry_pos_min} != align-table={self._row["qry_pos"]}'
        )

        assert (
                   qry_end_max := self.df_ops.select(pl.col('qry_end').max()).item()
               ) == self._row['qry_end'], (
            f'Query end position mismatch in LiftSeg init: lift-table={qry_end_max} != align-table={self._row["qry_end"]}'
        )

        assert (
                   ref_end_max := self.df_ops.select(pl.col('ref_end').max()).item()
               ) == self._row['end'], (
            f'Reference end position mismatch in LiftSeg init: lift-table={ref_end_max} != align-table={self._row["end"]}'
        )


    def __repr__(self) -> str:
        """Get a string representation of the object."""
        return (
            f'{self.__class__.__name__}'
            f'('
            f'{self.align_table_index}, '
            f'ref="{self._row["chrom"]}:{self._row["pos"]}-{self._row["end"]}", '
            f'qry="{self._row["qry_id"]}:{self._row["qry_pos"]}-{self._row["qry_end"]}"'
            f')'
        )

    def to_qry(
            self,
            pos: int
    ) -> tuple[int, int] | tuple[None, None]:
        df_ops = self.df_ops.filter(
            pl.col('ref_pos') <= pos,
            pl.col('ref_end') > pos,
        )

        if df_ops.height != 1:
            return None, None

        ref_pos, ref_end, qry_pos, qry_end, op_code = df_ops.select(
            'ref_pos', 'ref_end', 'qry_pos', 'qry_end', 'op_code'
        ).row(0)

        if qry_pos == qry_end:
            return qry_pos, op_code

        if self.is_rev:
            return (qry_end - 1) - (pos - ref_pos), op_code

        return qry_pos + (pos - ref_pos), op_code


    def to_ref(
            self,
            pos: int
    ) -> tuple[int, int] | tuple[None, None]:
        df_ops = self.df_ops.filter(
            pl.col('qry_pos') <= pos,
            pl.col('qry_end') > pos,
        )

        if df_ops.height != 1:
            return None, None

        ref_pos, ref_end, qry_pos, qry_end, op_code = df_ops.select(
            'ref_pos', 'ref_end', 'qry_pos', 'qry_end', 'op_code'
        ).row(0)

        if ref_pos == ref_end:
            return ref_pos, op_code

        if self.is_rev:
            return (ref_end - 1) - (pos - qry_pos), op_code

        return ref_pos + (pos - qry_pos), op_code

    def __getattr__(self, name: str) -> Any:
        if name == 'row':
            return self._row.copy()

        if name in self._row.keys():
            return self._row[name]

        raise AttributeError(f'Attribute {name} not found on {self.__class__.__name__}')


@immutable
class AlignLift:
    """Alignment liftover utility.

    Create an alignment liftover object for translating between reference and query alignments (both directions). Build
    liftover from alignment data in a DataFrame (requires chrom, pos, end, qry_id, qry_pos, and qry_end).

    :ivar df: Alignment DataFrame for retrieving alignment records. Column "_index" is added.
    :ivar df_col_set: Set of column names in df.
    :ivar df_index: Alignment DataFrame with a subset of df columns stored as a DataFrame.
        Column "_index" is added and matches "df".
    :ivar fai_dict: Query sequence lengths (key: query name, value: query sequence length).
    :ivar cache_align: Number of alignment records to cache liftover trees in memory.
    """
    df: pl.LazyFrame
    df_col_set: set[str]
    df_index: pl.DataFrame
    fai_dict: dict[str, int]
    cache_align: int = 10

    def __init__(
            self,
            df: pl.DataFrame | pl.LazyFrame,
            df_qry_fai: pl.DataFrame | pl.LazyFrame,
            cache_align: int = DEFAULT_CACHE_SIZE,
    ):
        """Initialize an alignment liftover object.

        :param df: Alignment DataFrame.
        :param df_qry_fai: Query FAI DataFrame.
        :param cache_align: Number of alignment records to cache in memory.
        """
        if df is None:
            raise ValueError(f'Parameter "df" is missing')

        df = (
            df
            .drop('_index', strict=False)
            .with_row_index('_index')
        )

        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        if df_qry_fai is None:
            raise ValueError(f'Parameter "df_qry_fai" is missing')

        if isinstance(df_qry_fai, pl.LazyFrame):
            df_qry_fai = df_qry_fai.collect()

        try:
            cache_align = max(int(cache_align), 1)
        except (ValueError, TypeError):
            cache_align = 10

        # Check alignment table
        expected_cols = {'_index', 'chrom', 'pos', 'end', 'qry_id', 'qry_pos', 'qry_end', 'is_rev', 'filter'}
        auto_cols = {'score', 'align_index', 'filter'}

        df_index = (
            df
            .select(cs.by_name(expected_cols, auto_cols, require_all=False))
            .collect()
        )

        if missing_cols := expected_cols - set(df_index.columns):
            raise ValueError(f'Missing alignment table columns: {", ".join(sorted(missing_cols))}')

        if 'align_index' not in df_index.columns:
            df_index = df_index.with_columns(
                pl.lit(None).cast(schema.ALIGN['align_index']).alias('align_index')
            )

        if 'score' not in df_index.columns:
            df_index = df_index.with_columns(
                pl.lit(0.0).cast(
                    FeatureGenerator.all_feature_schema()['score']
                )
            )

        if 'filter' not in df_index.columns:
            df_index = df_index.with_columns(
                pl.lit([]).cast(schema.ALIGN['filter']).alias('filter')
            )

        df_index = df_index.with_columns(
            pl.col('filter').fill_null([]).cast(schema.ALIGN['filter']),
        )

        if df_index.select(pl.col('align_index').is_duplicated().any()).item():
            raise ValueError('Duplicate "align_index" values found in alignment DataFrame')

        df_col_set = set(df.collect_schema().names())

        if 'align_ops' not in df_col_set:
            raise ValueError('Missing "align_ops" column in alignment table')

        # Assign fields
        self.df = df
        self.df_col_set = df_col_set
        self.df_index = df_index
        self._fai_dict = dict(zip(df_qry_fai['chrom'], df_qry_fai['len']))
        self.cache_align = cache_align

        self._get_lift_seg = lru_cache(maxsize=self.cache_align)(self._get_lift_seg_nocache)
        self._get_lift_seg.__doc__ = self._get_lift_seg_nocache.__doc__

    def to_ref(
            self,
            qry_id: str,
            qry_pos: int,
            op_code_filter: Optional[Iterable[int]] = None,
            align_pass: bool = True,
    ) -> list[dict[str, Any]]:
        """Lift coordinates from query to reference.

        Each dict has the following keys:
        - chrom: Reference chromosome
        - pos: Reference position in 0-based coordinates.
        - is_rev: Whether the query is reverse-complemented in this alignment
        - index: Location in the alignment table where the first record is 0 and each row
          increments by 1
        - align_index: Alignment align_index assigned when the alignment table was created
        - score: Alignment score
        - filter: Alignment filter flags
        - op_code: Alignment operation this lift fell into

        :param qry_id: Query record ID.
        :param qry_pos: Query location in 0-based coordinates.
        :param op_code_filter: Accept only lifts within these alignment operations.
        :param align_pass: Exclude alignments with a filter flag ("filter" column).

        :returns: A list of all possible lifts from `qry_id:qry_pos` to reference locations where
            each lift is a dict with keys listed above.
        """
        lift_list = []

        if op_code_filter is None:
            op_code_filter = {op.EQ, op.X, op.M, op.I, op.D}

        for row in self.df_index.filter(
            pl.col('qry_id') == qry_id,
            pl.col('qry_pos') <= qry_pos,
            pl.col('qry_end') > qry_pos,
        ).iter_rows(named=True):
            if not align_pass or len(row['filter']) > 0:
                continue

            lift_seg = self._get_lift_seg(row['_index'])

            ref_pos, op_code = lift_seg.to_ref(qry_pos)

            if ref_pos is not None and op_code in op_code_filter:
                lift_list.append(
                    {
                        'chrom': row['chrom'],
                        'pos': ref_pos,
                        'qry_id': qry_id,
                        'qry_pos': qry_pos,
                        'is_rev': lift_seg.is_rev,
                        'index': row['_index'],
                        'align_index': row['align_index'],
                        'score': row['score'],
                        'filter': row['filter'],
                        'op_code': op_code,
                    }
                )

        return lift_list

    def to_qry(
            self,
            chrom: str,
            pos: int,
            op_code_filter: Optional[Iterable[int]] = None,
            align_pass: bool = True,
    ) -> list[dict[str, Any]]:
        """Lift coordinates from reference to query.

        Each dict has the following keys:
        - chrom: Query ID
        - pos: Query position in 0-based coordinates
        - is_rev: Whether the query is reverse-complemented
        - index: Location in the alignment table where the first record is 0 and each row
          increments by 1
        - align_index: Alignment align_index assigned when the alignment table was created
        - score: Alignment score
        - filter: Alignment filter flags
        - op_code: Alignment operation this lift fell into

        :param chrom: Reference chromosome ID.
        :param pos: Reference location in 0-based coordinates.
        :param op_code_filter: Accept only lifts within these alignment operations.
        :param align_pass: Exclude alignments with a filter flag ("filter" column).

        :returns: A list of all possible lifts from `chrom:pos` to query locations where each lift
            is a dict with keys listed above.
        """
        lift_list = []

        if op_code_filter is None:
            op_code_filter = {op.EQ, op.X, op.M, op.I, op.D}

        for row in self.df_index.filter(
            pl.col('chrom') == chrom,
            pl.col('pos') <= pos,
            pl.col('end') > pos,
        ).iter_rows(named=True):
            if not align_pass or len(row['filter']) > 0:
                continue

            lift_seg = self._get_lift_seg(row['_index'])

            qry_pos, op_code = lift_seg.to_qry(pos)

            if qry_pos is not None and op_code in op_code_filter:
                lift_list.append(
                    {
                        'chrom': chrom,
                        'pos': pos,
                        'qry_id': row['qry_id'],
                        'qry_pos': qry_pos,
                        'is_rev': lift_seg.is_rev,
                        'index': row['_index'],
                        'align_index': row['align_index'],
                        'score': row['score'],
                        'filter': row['filter'],
                        'op_code': op_code,
                    }
                )

        return lift_list

    @property
    def fai_dict(self) -> dict[str, int]:
        """Get a dict mapping query IDs to query sequence lengths."""
        return self._fai_dict.copy()


    # def region_to_ref(
    #         self,
    #         region_qry: Region,
    #         same_index: bool = False
    # ):
    #     """Lift region to reference.
    #
    #     :param region_qry: Query region.
    #     :param same_index: If True, the region must have the same alignment index as the reference region.
    #
    #     :returns: Reference region or `None` if the query region could not be lifted.
    #     """
    #     # Lift
    #     ref_pos, end_ref = self.coord_to_ref(region_qry.chrom, (region_qry.pos, region_qry.end))
    #
    #     if same_index:
    #         if region_qry.pos_align_index is not None:
    #             ref_pos = [_ for _ in ref_pos if _['align_index'] == region_qry.pos_align_index]
    #
    #         if region_qry.end_align_index is not None:
    #             end_ref = [_ for _ in end_ref if _['align_index'] == region_qry.end_align_index]
    #
    #     if len(ref_pos) != 1 or len(end_ref) != 1:
    #         return None
    #
    #     ref_pos = ref_pos[0]
    #     end_ref = end_ref[0]
    #
    #     if ref_pos['chrom'] != end_ref['chrom'] or ref_pos['is_rev'] != end_ref['is_rev']:
    #         return None
    #
    #     # Return
    #     return Region(
    #         chrom=ref_pos['chrom'],
    #         pos=min(ref_pos['pos'], end_ref['pos']),
    #         end=max(ref_pos['pos'], end_ref['pos']),
    #         is_rev=ref_pos['is_rev'],
    #         pos_align_index=ref_pos['align_index'],
    #         end_align_index=end_ref['align_index'],
    #     )
    #
    # def region_to_qry(
    #         self,
    #         region_ref: Region,
    #         same_index: bool = False
    # ):
    #     """Lift region to query.
    #
    #     :param region_ref: Reference region.
    #     :param same_index: If True, the region must have the same alignment index as the reference region.
    #
    #     :returns: Query region or `None` if it could not be lifted.
    #     """
    #     # Lift
    #     query_pos, query_end = self.coord_to_qry(region_ref.chrom, (region_ref.pos, region_ref.end))
    #
    #     if same_index:
    #         if region_ref.pos_align_index is not None:
    #             query_pos = [_ for _ in query_pos if _['align_index'] == region_ref.pos_align_index]
    #
    #         if region_ref.end_align_index is not None:
    #             query_end = [_ for _ in query_end if _['align_index'] == region_ref.end_align_index]
    #
    #     if len(query_pos) != 1 or len(query_end) != 1:
    #         return None
    #
    #     query_pos = query_pos[0]
    #     query_end = query_end[0]
    #
    #     if query_pos['chrom'] != query_end['chrom'] or query_pos['is_rev'] != query_end['is_rev']:
    #         return None
    #
    #     # Return
    #     return Region(
    #         chrom=query_pos['chrom'],
    #         pos=min(query_pos['pos'], query_end['pos']),
    #         end=max(query_pos['pos'], query_end['pos']),
    #         is_rev=query_pos['is_rev'],
    #         pos_align_index=query_pos['align_index'],
    #         end_align_index=query_end['align_index'],
    #     )

    def _get_lift_seg_nocache(
            self,
            align_table_index: int,
    ) -> LiftSeg:
        """Generate a lift segment for a given alignment table index.

        :param align_table_index: Alignment table index.

        :returns: Lift segment.

        :raises ValueError: If the alignment table index is not found.
        """
        return LiftSeg(
            align_table_index,
            self.df,
            self.df_index,
            self._fai_dict,
        )

    def __repr__(self):
        """Get a string representation."""
        return 'AlignLift()'

def _get_df_op(
        align_table_index: int,
        df: pl.LazyFrame,
        qry_len: int,
) -> tuple[pl.DataFrame, bool]:
    """Get tables of operations for a given alignment table index.

    Three items are returned:

    1. Table of operations for lifting from reference to query
    2. Table of operations for lifting from query to reference
    3. Boolean indicating if the alignment is reverse complemented

    :param align_table_index: Row number in `df` containing this alignment.
    :param df: Table of alignment records.
    :param qry_len: Length of the query sequence.

    :returns: Tuple of two tables (to-query and to-reference) and a boolean indicating if the
        alignment is reverse complemented.
    """
    df_ops = df.filter(pl.col('_index') == align_table_index)

    assert df_ops.select(pl.len()).collect().item() == 1, (
        f'Alignment record at row {align_table_index} not found or not unique: '
        f'Found {df_ops.select(pl.len()).collect().item()} records'
    )

    is_rev, ref_pos = df_ops.select('is_rev', 'pos').collect().row(0)

    df_ops = (
        df_ops
        .select(
            pl.col('align_ops').struct.field('op_code'),
            pl.col('align_ops').struct.field('op_len'),
        )
        .explode('op_code', 'op_len')
        .select(
            'op_code', 'op_len',
            pl.when(pl.col('op_code').is_in(op.ADV_QRY_ARR))
            .then(pl.col('op_len'))
            .otherwise(0)
            .alias('qry_len'),
            pl.when(pl.col('op_code').is_in(op.ADV_REF_ARR))
            .then(pl.col('op_len'))
            .otherwise(0)
            .alias('ref_len'),
        )
        .select(
            'op_code', 'op_len', 'ref_len', 'qry_len',
            (pl.col('ref_len').cum_sum() + ref_pos).alias('ref_end'),
            (pl.col('qry_len').cum_sum()).alias('qry_end'),
        )
        .select(
            'op_code', 'op_len',
            pl.col('ref_end').shift(1, fill_value=ref_pos).alias('ref_pos'),
            'ref_end',
            pl.col('qry_end').shift(1, fill_value=0).alias('qry_pos'),
            'qry_end',
        )
    )

    if is_rev:
        df_ops = (
            df_ops
            .with_columns(
                (qry_len - (pl.col('qry_end'))).alias('qry_pos'),
                (qry_len - pl.col('qry_pos')).alias('qry_end'),
            )
        )

    return (
        df_ops
        .filter(pl.col('op_code').is_in([op.M, op.I, op.D, op.EQ, op.X]))
        .collect(),
        is_rev
    )
