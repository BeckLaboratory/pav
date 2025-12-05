"""Create and manaage alignemnt BED files."""

__all__ = [
    'ALIGN_TABLE_SORT_ORDER',
    'NAMED_COORD_COLS',
    'DEPTH_SCHEMA',
    'sam_to_align_table',
    'align_depth_table',
    'align_depth_filter',
    'intersect_other',
    'qry_order_expr',
    'align_stats',
    'check_table',
]

import numpy as np
import os
import polars as pl
from typing import Iterable, Optional

from .. import schema

from ..io import SamStreamer

from . import op

from .features import FeatureGenerator
from .lcmodel import LCAlignModel, null_model
from .records import check_record
from .score import ScoreModel, get_score_model


ALIGN_TABLE_SORT_ORDER = ['chrom', 'pos', 'end', 'align_index']
"""Sort order for alignment tables."""

NAMED_COORD_COLS = {
    'ref': ('chrom', 'pos', 'end'),
    'qry': ('qry_id', 'qry_pos', 'qry_end')
}
"""
Named coordinate columns for alignment table depths.

Maps a string alias to a tuple of column names. Simplifies depth across reference or query sequences.
"""

DEPTH_SCHEMA = {
    'chrom': pl.String,
    'pos': schema.ALIGN['pos'],
    'end': schema.ALIGN['end'],
    'depth': pl.Int32,
    'index': pl.List(schema.ALIGN['align_index']),
}
"""Schema for alignment depth tables."""


def sam_to_align_table(
        sam_filename: str,
        df_qry_fai: pl.DataFrame,
        min_mapq: int = 0,
        score_model: Optional[ScoreModel | str] = None,
        lc_model: Optional[LCAlignModel] = None,
        align_features: Optional[Iterable[str] | str] = 'align',
        flag_filter: int = 0x700,
        ref_fa_filename: Optional[str] = None
) -> pl.DataFrame:
    """Read alignment records from a SAM file.

    Avoid pysam, it uses htslib, which has a limit of 268,435,456 bp for each alignment record, and clipping on a CIGAR
    string can exceed this limit (https://github.com/samtools/samtools/issues/1667) and causing PAV to crash with an
    error message starting with "CIGAR length too long at position".

    :param sam_filename: File to read.
    :param df_qry_fai: Pandas Series with query names as keys and query lengths as values.
    :param min_mapq: Minimum MAPQ score for alignment record.
    :param score_model: Score model to use.
    :param lc_model: LCAlignModel to use.
    :param align_features: List of alignment features to add to the alignment table. May be an iterable of feature names
        or a single string indicating a named feature set ("align_table" is the default  features for alignment
        tables, "all" is all known features). If None, use "align_table".
    :param flag_filter: Filter alignments matching these flags.
    :param ref_fa_filename: Reference FASTA filename.

    :returns: Table of alignment records.

    :raises ValueError: If function arguments are invalid.
    """
    if lc_model is None:
        lc_model = null_model()

    # Rename chrom to qry_id in FAI if it wasn't already done
    if 'chrom' in df_qry_fai.columns:
        df_qry_fai = df_qry_fai.rename({'chrom': 'qry_id'})

    # Get score model and feature generator
    score_model = get_score_model(score_model)

    feature_gen = FeatureGenerator(
        features=align_features,
        score_model=score_model,
        force_all=True  # Not necessary, but overwrite features if already present
    )

    if conflict_set := set(feature_gen.features) & set(schema.ALIGN.keys()):
        raise ValueError(
            f'Feature names conflict with standard alignment table columns: {", ".join(sorted(conflict_set))}'
        )

    columns_head = [
        'chrom', 'pos', 'end',
        'align_index',
        'filter',
        'qry_id', 'qry_pos', 'qry_end',
        'qry_order',
        'rg',
        'mapq',
        'is_rev', 'flags',
        'align_ops',
    ]

    if not os.path.isfile(sam_filename) or os.stat(sam_filename).st_size == 0:
        raise FileNotFoundError('SAM file is empty or missing')

    # Get records from SAM
    record_list = list()

    align_index = -1
    line_number = 0

    with SamStreamer(sam_filename, ref_fa=ref_fa_filename) as in_file:
        for line in in_file:
            line_number += 1

            try:

                line = line.strip()

                if line.startswith('@') or not line:
                    continue

                align_index += 1

                tok = line.split('\t')

                if len(tok) < 11:
                    raise ValueError(f'Expected at least 11 fields, received {len(tok)} at line {line_number}')

                # Note: values are prefixed with type and colon, (e.g. {"NM": "i:579204"}).
                tag = dict(val.split(':', 1) for val in tok[11:])

                if 'CG' in tag:
                    raise ValueError('Found BAM-only tag "CG"')

                if 'RG' in tag:
                    if not tag['RG'].startswith('Z:'):
                        raise ValueError(f'Found non-Z RG tag: {tag["RG"]}')
                    tag_rg = tag['RG'][2:].strip()

                    if not tag_rg:
                        tag_rg = None

                else:
                    tag_rg = None

                flags = int(tok[1])
                mapq = int(tok[4])
                is_rev = bool(flags & 0x10)

                pos_ref = int(tok[3]) - 1

                # Skipped unmapped reads, low MAPQ reads, or other flag-based filters
                if flags & 0x4 or mapq < min_mapq or pos_ref < 0:
                    continue

                # Get alignment operations
                op_arr = op.clip_soft_to_hard(op.cigar_to_arr(tok[5]))

                if np.any(op_arr[:, 0] * op.M):
                    raise ValueError('PAV does not allow match alignment operations (op "M", requires "=" and "X")')

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
                    raise ValueError(f'Found mapped read with missing names (chrom={chrom}, qry_id={qry_id})')

                # Save record
                row = {
                    'chrom': chrom,
                    'pos': pos_ref,
                    'end': pos_ref + len_ref,
                    'align_index': align_index,
                    'filter': [] if not flags & flag_filter else ['ALIGN'],
                    'qry_id': qry_id,
                    'qry_pos': pos_qry,
                    'qry_end': pos_qry + len_qry,
                    'qry_order': None,
                    'tag_rg': tag_rg,
                    'mapq': mapq,
                    'is_rev': is_rev,
                    'flags': flags,
                    'align_ops': op.arr_to_row(op_arr)
                }

                record_list.append(row)

            except Exception as e:
                raise ValueError('Failed to parse record at line {}: {}'.format(line_number, str(e))) from e

    # Merge records
    df = pl.DataFrame(record_list, orient='row', schema=schema.ALIGN)

    df = (
        df
        .with_columns(qry_order_expr())
        .select(columns_head)
    )

    # Compute features
    df = feature_gen(df, df_qry_fai)

    # Set filter
    filter_loc = lc_model(
        df,
        existing_score_model=score_model,
        df_qry_fai=df_qry_fai
    )

    df = (
        df
        .with_columns(
            pl.when(pl.Series(filter_loc))
            .then(pl.col('filter').list.concat(pl.Series(['LCALIGN'])))
            .otherwise(pl.col('filter'))
            .alias('filter')
        )
    )

    # Reference order
    df = df.sort(ALIGN_TABLE_SORT_ORDER)

    # Check sanity
    for row in df.iter_rows(named=True):
        check_record(row, df_qry_fai)

    check_table(df)

    # Return alignment table
    return df


def align_depth_table(
        df: pl.DataFrame | pl.LazyFrame,
        df_fai: Optional[pl.DataFrame | pl.LazyFrame] = None,
        coord_cols: Iterable[str] | str = ('chrom', 'pos', 'end'),
        retain_filtered: bool = True
) -> pl.DataFrame:
    """Get a table of alignment depth from an alignment table.

    Table columns:
        - chrom or qry_id: Chromosome name
        - pos or qry_pos: Start position.
        - end or qry_end: End position.
        - depth: Depth of alignments between pos and end.

    The first three columns are derived from argument `coord_cols`.

    :param df: Alignment table.
    :param df_fai: Reference FASTA index table. Must have a column named "len" and a column matching the first element
        of `coord_cols`.
    :param coord_cols: Coordinate columns to use for depth. Typically ('chrom', 'pos', 'end') or
        ('qry_id', 'qry_pos', 'qry_end'). If a string, must be "ref" (for chrom, pos, end) or
        "qry" (for qry_id, qry_pos, qry_end).
    :param retain_filtered: Retain filtered alignments if True.

    :returns: Depth table.

    :raises ValueError: If columns are missing from tables df or df_fai.
    """
    if not isinstance(df, pl.LazyFrame):
        df = df.lazy()

    # Check arguments
    coord_cols = _check_coord_cols(coord_cols)

    if (missing_cols := set(coord_cols) - set(df.collect_schema().names())) != set():
        raise ValueError(
            f'coord_cols must be a subset of df columns. Missing columns: {", ".join(sorted(missing_cols))}'
        )

    if df_fai is not None:
        if not isinstance(df_fai, pl.DataFrame):
            df_fai = df_fai.collect()

        if coord_cols[0] not in df_fai.collect_schema().names():
            raise ValueError(f'coord_cols[0] must be in df_fai columns: {coord_cols[0]}')

        if 'len' not in df_fai.collect_schema().names():
            raise ValueError('df_fai must have a "len" column')

        df_fai = df_fai.select([coord_cols[0], 'len'])

    col_chrom = pl.col(coord_cols[0])
    col_pos = pl.col(coord_cols[1])
    col_end = pl.col(coord_cols[2])
    col_filter = pl.col('filter')

    # Prepare alignment table
    df_cols = df.collect_schema().names()

    if 'filter' not in df_cols:
        df = df.with_columns(pl.lit([]).alias('filter'))

    df = (
        df.select(col_chrom, col_pos, col_end, col_filter)
        .with_row_index('index')
    )

    if not retain_filtered:
        df = df.filter(pl.col('filter').list.len() == 0)

    df = df.filter(col_end > col_pos)

    # Get depth per chromosome
    chrom_list = df.select(col_chrom).unique().collect().to_series().sort().to_list()

    df_depth_list = []

    for chrom in chrom_list:
        df_depth = (
            df
            .filter(col_chrom == chrom)
            .select(
                pl.concat_list([
                    pl.struct([col_pos.alias('coord'), pl.lit(1).alias('dir')]),
                    pl.struct([col_end.alias('coord'), pl.lit(-1).alias('dir')])
                ]).alias('coord_pair'),
                pl.col('index')
            )
            .explode('coord_pair')
            .unnest('coord_pair')
            .sort(['coord', 'dir'])
            .select(
                pl.col('coord'),
                pl.col('dir').cum_sum().alias('depth'),
                pl.col('index')
            )
            .select(
                pl.col('coord').alias('pos'),
                (
                    pl.col('coord').shift(-1, fill_value=(
                        df_fai.row(by_predicate=col_chrom == chrom, named=True)['len']
                        if df_fai is not None else pl.col('coord').max()
                    ))
                ).alias('end'),
                pl.col('depth'),
                pl.col('index')
            )
            .collect()
        )

        # Get indexes
        index_col_list = [[]]
        last_depth = 0

        for depth, index in df_depth.select(['depth', 'index']).rows():
            assert last_depth != depth

            last_index_list = index_col_list[-1]

            if depth > last_depth:
                assert index not in last_index_list

                index_col_list.append(
                    last_index_list.copy() + [index]
                )
            else:
                assert index in last_index_list

                index_col_list.append(
                    [i for i in last_index_list if i != index]
                )

            last_depth = depth

        index_col_list = index_col_list[1:]

        df_depth = (
            df_depth
            .with_columns(
                pl.Series(index_col_list, dtype=pl.List(pl.Int64)).alias('index')
            )
            .filter(
                pl.col('pos') < pl.col('end')
            )
            .select(
                pl.lit(chrom).alias('chrom'),
                pl.col('pos'),
                pl.col('end'),
                pl.col('depth'),
                pl.col('index')
            )
            .cast(DEPTH_SCHEMA)
        )

        # Ends
        if df_fai is not None and (min_pos := df_depth.select(pl.col('pos').min())['pos'][0]) > 0:
            df_depth = (
                pl.concat([
                    pl.DataFrame({
                        'chrom': [chrom],
                        'pos': [0],
                        'end': [min_pos],
                        'depth': [0],
                        'index': [[]]
                    }, schema=DEPTH_SCHEMA),
                    df_depth
                ])
                .select(df_depth.columns)
            )
        else:
            df_depth = df_depth.filter(pl.col('depth') > 0)

        df_depth_list.append(df_depth)

    # Concat
    df_depth = (
        pl.concat(df_depth_list) if len(df_depth_list) > 0 else pl.DataFrame(schema=DEPTH_SCHEMA)
        .rename({
            'chrom': coord_cols[0],
            'pos': coord_cols[1],
            'end': coord_cols[2],
        })
        .sort(['chrom', 'pos', 'end'])
    )

    # Return
    return df_depth


def align_depth_filter(
        df: pl.DataFrame,
        df_depth: Optional[pl.DataFrame],
        max_depth: int,
        max_overlap: float,
        coord_cols: Iterable[str] | str = ('chrom', 'pos', 'end'),
        append_filter: str = 'DEPTH'
) -> pl.DataFrame:
    """Filter alignments based on overlap with deeply-mapped regions.

    Intersect df with df_depth and count the number of bases in df overlapping all records in df_depth (may overlap
    multiple records, intersect bases are summed). If the proportion of intersected bases exceeds max_overlap, append
    "append_filter" to the "filter" column in df. A reasonable max_overlap threshold will permit long alignments to
    pass through deeply-mapped regions without being filtered.

    :param df: Table of alignment records.
    :param df_depth: Table of depth records. If None, will be generated from df.
    :param max_depth: Maximum depth, filter records intersecting loci exceeding this depth (> max_depth).
    :param max_overlap: Maximum overlap allowed when intersecting deep alignment regions.
    :param coord_cols: Column names for coordinates (chrom, start, end).
    :param append_filter: String to append to "filter" column in df.

    :returns df with "filter" column appended.

    :raises ValueError: Arguments are invalid.
    """
    # Check arguments
    if df is None:
        raise ValueError('df must be specified')

    if append_filter is None or (append_filter := str(append_filter).strip()) == '':
        raise ValueError('Missing append_filter')

    # Prepare tables
    if df_depth is None:
        df_depth = align_depth_table(df, coord_cols=coord_cols)

    if (missing_cols := set(coord_cols) - set(df.columns)) != set():
        raise ValueError(
            f'Missing columns in df: {", ".join(sorted(missing_cols))}'
        )

    if (missing_cols := (set(coord_cols) | {'depth'}) - set(df_depth.columns)) != set():
        raise ValueError(
            f'Missing columns in df_depth: {", ".join(sorted(missing_cols))}'
        )

    if 'filter' not in df.columns:
        df = df.with_columns(pl.lit([]).cast(pl.List(pl.String)).alias('filter'))

    # Get intersects
    filter_index = (
        intersect_other(
            df, df_depth.filter(pl.col('depth') > max_depth),
            coord_cols=coord_cols
        )
        .filter(pl.col('bp_prop') > max_overlap)
        .select(pl.col('index'))
        .to_series()
        .to_list()
    )

    df = (
        df
        .with_row_index('_index')
        .with_columns(
            pl.when(
                pl.col('_index').is_in(filter_index) & ~ pl.col('filter').list.contains(append_filter)
            )
            .then(pl.col('filter').list.concat(pl.lit([append_filter])))
            .otherwise(pl.col('filter'))
            .alias('filter')
        )
        .drop('_index')
    )

    return df


def intersect_other(
        df: pl.DataFrame,
        df_other: pl.DataFrame,
        coord_cols: Iterable[str] | str = ('chrom', 'pos', 'end')
) -> pl.DataFrame:
    """Intersect tables by coordinates and count the number of overlapping bases.

    .. Warning::
       If df_other contains overlapping records, the intersect bases will be counted multiple times. This function
       is typically used with a depth table, which does not contain overlaps.

    The returned table has the following fields:

        * index: Index in df by position (first record is 0, secord is 1, etc).
        * len: Length of the record (end position - start position).
        * bp: Number of bases in df overlapping all records in df_other.
        * bp_prop: Proportion of bases in df overlapping all records in df_other.

    :param df: Table with coordinates.
    :param df_other: Other table with coordinates.
    :param coord_cols: Coordinate columns to use for depth. Typically ('chrom', 'pos', 'end') or
        ('qry_id', 'qry_pos', 'qry_end'). If a string, must be "ref" (for chrom, pos, end) or
        "qry" (for qry_id, qry_pos, qry_end).

    :returns: An intersect table.

    :raises ValueError: Arguments are invalid.
    """
    # Check arguments
    if df is None:
        raise ValueError('df must be specified')

    if df_other is None:
        raise ValueError('df_other must be specified')

    coord_cols = _check_coord_cols(coord_cols)

    # Prepare tables
    if (missing_cols := set(coord_cols) - set(df.columns)) != set():
        raise ValueError(
            f'coord_cols must be a subset of df columns. Missing columns: {", ".join(sorted(missing_cols))}'
        )

    if (missing_cols := set(coord_cols) - set(df_other.columns)) != set():
        raise ValueError(
            f'coord_cols must be a subset of df_depth columns. Missing columns: {", ".join(sorted(missing_cols))}'
        )

    col_chrom = pl.col(coord_cols[0])
    col_pos = pl.col(coord_cols[1])
    col_end = pl.col(coord_cols[2])

    chrom_list = df.select(col_chrom).unique().to_series().sort().to_list()

    df = df.with_row_index('_index')

    df_coord = (
        df.lazy()
        .select(
            col_chrom.alias('chrom'),
            col_pos.alias('pos_a'),
            col_end.alias('end_a'),
            pl.col('_index').alias('index')
        )
    )

    df_other = (
        df_other.lazy()
        .select(
            col_chrom.alias('chrom'),
            col_pos.alias('pos_b'),
            col_end.alias('end_b')
        )
    )

    df_len = (  # Length of each alignment (key=index, value=len)
        df_coord
        .select(
            pl.col('index'),
            (pl.col('end_a') - pl.col('pos_a')).alias('len')
        )
    )

    # Filter
    df_list = list()

    for chrom in chrom_list:

        df_list.append(
            df_coord.filter(pl.col('chrom') == chrom)
            .join_where(
                df_other.filter(pl.col('chrom') == chrom),
                pl.col('pos_a') < pl.col('end_b'),
                pl.col('end_a') > pl.col('pos_b')
            )
            .group_by('index')
            .agg(
                (
                    pl.min_horizontal(
                        pl.col('end_a'), pl.col('end_b')
                    ) - pl.max_horizontal(
                        pl.col('pos_a'), pl.col('pos_b')
                    )
                ).sum().alias('bp')
            )
            .join(
                df_len,
                on='index',
                how='left'
            )
            .select(
                pl.col('index'),
                pl.col('bp'),
                (pl.col('bp') / pl.col('len')).alias('bp_prop')
            )
        )

    return (
        df_len
        .join(
            pl.concat(df_list),
            on='index',
            how='left'
        )
        .with_columns(
            pl.col('bp').fill_null(0),
            pl.col('bp_prop').fill_null(0.0)
        )
    ).collect()


def _check_coord_cols(
        coord_cols: Iterable[str] | str
) -> tuple[str, str, str]:
    """Check coord_cols and substitute defaults.

    Convenience method for checking coord_cols parameters used by several functions.

    :param coord_cols: A tuple of three elements are a pre-defined keyword indicating which coordinate columns to use.

    :returns: A a tuple af three column names, e.g. ('chrom', 'pos', 'end').
    """
    if coord_cols is None:
        raise ValueError('coord_cols must be specified')

    if isinstance(coord_cols, str):
        coord_cols: tuple[str, str, str] = NAMED_COORD_COLS.get(coord_cols, None)

        if coord_cols is None:
            raise ValueError('If coord_cols is a string, it must be "ref" or "qry"')

    coord_cols = tuple(coord_cols)

    if len(coord_cols) != 3:
        raise ValueError(
            f'coord_cols must have length 3 [(chrom, pos, end) or (qry_id, qry_pos, qry_end)), length={len(coord_cols)}'
        )

    return coord_cols


def qry_order_expr() -> pl.Expr:
    """Get an expression for computing the query order of a table.

    For any query sequence, the first alignment record in the sequence (i.e. containing the left-most aligned base
    relative to the query sequence) will have order 0, the next alignment record 1, etc. The order is set per query
    sequence (i.e. the first aligned record of every unique query ID will have order 0).

    :returns: An expression for computing the query order.
    """
    return (
        (
            pl.struct(['qry_pos', 'qry_end'])
            .rank(method='ordinal')
            .over('qry_id')
            - 1
        )
        .alias('qry_order')
    )


def align_stats(
        df: pl.DataFrame,
        head_cols: Optional[list[tuple[str, str]]] = None
) -> pl.DataFrame:
    """Collect high-level stats from an alignment table separated by by filter.

    Statis are divided by passing filters (filter list is empty) and fail (filter list is not empty).

    :param df: Alignment table.
    :param head_cols: List of tuples of (column name, literal value) to prepend to the output table.

    :returns: A table with new or updated feature columns.
    """
    agg_list = [
        # n
        pl.len().alias('n'),

        # n_prop
        (pl.len() / df.height).alias('n_prop'),

        # bp
        (pl.col('qry_end') - pl.col('qry_pos')).sum().alias('bp'),

        # bp_prop
        (
            (pl.col('qry_end') - pl.col('qry_pos')).sum() / (df['qry_end'] - df['qry_pos']).sum()
        ).alias('bp_prop'),

        # bp_mean
        ((pl.col('qry_end') - pl.col('qry_pos')).sum() / pl.len()).alias('bp_mean'),

        # bp_ref
        (pl.col('end') - pl.col('pos')).sum().alias('bp_ref'),

        # bp_ref_prop
        (
            (pl.col('end') - pl.col('pos')).sum() / (df['end'] - df['pos']).sum()
        ).alias('bp_ref_prop'),

        # bp_ref_mean
        ((pl.col('end') - pl.col('pos')).sum() / pl.len()).alias('bp_ref_mean'),

        # mapq_mean
        pl.col('mapq').mean().alias('mapq_mean')
    ]

    if 'filter' not in df.columns:
        df = df.with_columns(pl.lit([]).alias('filter'))

    if 'score' in df.columns:
        agg_list.append(pl.col('score').mean().alias('score_mean'))

    if 'score_prop' in df.columns:
        agg_list.append(pl.col('score_prop').mean().alias('score_prop_mean'))

    if 'score_mm' in df.columns:
        agg_list.append(pl.col('score_mm').mean().alias('score_mm_mean'))

    if 'score_mm_prop' in df.columns:
        agg_list.append(pl.col('score_mm_prop').mean().alias('score_mm_prop_mean'))

    if 'match_prop' in df.columns:
        agg_list.append(pl.col('match_prop').mean().alias('match_prop_mean'))

    df_sum = (
        df.lazy()
        .group_by('filter')
        .agg(*agg_list)
        .sort('filter')
        .collect()
    )

    if head_cols is not None:
        df_sum = (
            df_sum
            .select(
                *[
                    pl.lit(lit).alias(col)
                    for col, lit in head_cols
                ],
                pl.col('*')
            )
        )

    return df_sum


def check_table(df: pl.LazyFrame | pl.DataFrame) -> None:
    """Check alignment table invariants.

    Raises exceptions if tests do not pass.

    :param df: Alignment table to check.

    :raises ValueError: If errors are found in `df`.
    """

    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    # Duplicated indexes
    dup_index = (
        df
        .group_by('align_index')
        .agg(pl.len().alias('count'))
        .filter(pl.col('count') > 1)
        .sort('count', descending=True)
        .select(['align_index', 'count'])
        .collect().rows()
    )

    if dup_index:
        n = len(dup_index)
        dup_index = ', '.join(
            [f'{align_index} (n={count})' for align_index, count in dup_index]
        ) + ('...' if n > 3 else '')

        raise ValueError(f'Alignment table contains {n} duplicated record ID(s): {dup_index}')

    # Null values
    null_vals = (
        df
        .select(

            pl.col('chrom').is_null().sum(),
            pl.col('pos').is_null().sum(),
            pl.col('end').is_null().sum(),
            pl.col('align_index').is_null().sum(),
            pl.col('filter').is_null().sum(),
            pl.col('qry_id').is_null().sum(),
            pl.col('qry_pos').is_null().sum(),
            pl.col('qry_end').is_null().sum(),
            pl.col('qry_order').is_null().sum(),
            pl.col('is_rev').is_null().sum(),
            pl.col('score').is_null().sum(),
        )
        .collect()
        .transpose(include_header=True, header_name='col', column_names=['count'])
        .filter(pl.col('count') > 0)
        .rows()
    )

    if null_vals:
        n = len(null_vals)
        null_vals = ', '.join(
            [f'{col} (n={count})' for col, count in null_vals]
        ) + ('...' if n > 3 else '')

        raise ValueError(f'Alignment table contains {n} column(s) with null values: {null_vals}')
