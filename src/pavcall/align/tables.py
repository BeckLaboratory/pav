"""
Create and manaage alignemnt BED files.
"""

import numpy as np
import os
import polars as pl
from typing import Iterable, Optional

from .. import io
from .. import schema

from . import features
from . import lcmodel
from . import op
from . import records
from . import score

ALIGN_TABLE_SORT_ORDER = ['chrom', 'pos', 'end', 'align_index']
"""list[str]: Sort order for alignment tables."""

NAMED_COORD_COLS = {
    'ref': ('chrom', 'pos', 'end'),
    'qry': ('qry_id', 'qry_pos', 'qry_end')
}
"""
dict[str, tuple[str, str, str]]: Named coordinate columns for alignment table depths. Maps a string alias to a tuple
of column names. Simplifies depth across reference or query sequences.
"""

def sam_to_align_table(
        sam_filename: str,
        df_qry_fai: pl.DataFrame,
        min_mapq: int = 0,
        score_model: Optional[score.ScoreModel | str] = None,
        lc_model: Optional[lcmodel.LCAlignModel] = None,
        align_features: Optional[Iterable[str] | str] = 'align',
        flag_filter: int = 0x700,
        ref_fa_filename: Optional[str] = None
) -> pl.DataFrame:
    """
    Read alignment records from a SAM file. Avoid pysam, it uses htslib, which has a limit of 268,435,456 bp for each
    alignment record, and clipping on a CIGAR string can exceed this limit
    (https://github.com/samtools/samtools/issues/1667) and causing PAV to crash with an error message starting with
    "CIGAR length too long at position".

    Args:
        sam_filename: File to read.
        df_qry_fai: Pandas Series with query names as keys and query lengths as values.
        min_mapq: Minimum MAPQ score for alignment record.
        score_model: Score model to use.
        lc_model: LCAlignModel to use.
        align_features: List of alignment features to add to the alignment table. May be an iterable of feature names
            or a single string indicating a named feature set ("align_table" is the default  features for alignment
            tables, "all" is all known features). If None, use "align_table".
        flag_filter: Filter alignments matching these flags.
        ref_fa_filename: Reference FASTA filename.

    Returns:
        Table of alignment records.

    Raises:
        ValueError: If function arguments are invalid.
    """

    if lc_model is None:
        lc_model = lcmodel.null_model()

    # Rename chrom to qry_id in FAI if it wasn't already done
    if 'chrom' in df_qry_fai.columns:
        df_qry_fai = df_qry_fai.rename({'chrom': 'qry_id'})

    # Get score model and feature generator
    score_model = score.get_score_model(score_model)

    feature_gen = features.FeatureGenerator(
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

    with io.SamStreamer(sam_filename, ref_fa=ref_fa_filename) as in_file:
        for line in in_file:
            line_number += 1

            try:

                line = line.strip()

                if line.startswith('@') or not line:
                    continue

                align_index += 1

                tok = line.split('\t')

                if len(tok) < 11:
                    raise RuntimeError('Expected at least 11 fields, received {}'.format(line_number, len(tok)))

                tag = dict(val.split(':', 1) for val in tok[11:])  # Note: values are prefixed with type and colon, (e.g. {"NM": "i:579204"}).

                if 'CG' in tag:
                    raise RuntimeError(f'Found BAM-only tag "CG"')

                if 'RG' in tag:
                    if not tag['RG'].startswith('Z:'):
                        raise RuntimeError(f'Found non-Z RG tag: {tag["RG"]}')
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
                raise RuntimeError('Failed to parse record at line {}: {}'.format(line_number, str(e))) from e

    # Merge records
    df = pl.DataFrame(record_list, orient='row', schema=schema.ALIGN)

    df = (
        df
        .with_columns(qry_order=get_qry_order(df))
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
        records.check_record(row, df_qry_fai)

    # Return alignment table
    return df


def align_depth_table(
        df: pl.DataFrame | pl.LazyFrame,
        df_fai: Optional[pl.DataFrame | pl.LazyFrame] = None,
        coord_cols: Iterable[str] | str = ('chrom', 'pos', 'end'),
        retain_filtered: bool = True
) -> pl.DataFrame:
    """
    Get a table of alignment depth from an alignment table.

    Table columns:
        chrom or qry_id: Chromosome name
        pos or qry_pos: Start position.
        end or qry_end: End position.
        depth: Depth of alignments between pos and end.

    The first three columns are derived from coord_cols.

    Args:
        df: Alignment table.
        df_fai: Reference FASTA index table. Must have a column named "len" and a column matching the first element of
            coord_cols.
        coord_cols: Coordinate columns to use for depth. Typically ('chrom', 'pos', 'end') or
            ('qry_id', 'qry_pos', 'qry_end'). If a string, must be "ref" (for chrom, pos, end) or
            "qry" (for qry_id, qry_pos, qry_end).
        retain_filtered: Retain filtered alignments if True.

    Returns:
        Depth table.

    Raises:
        ValueError: If columns are missing from tables df or df_fai.
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
            raise ValueError(f'coord_cols[0] must be in df_fai columns')

        if 'len' not in df_fai.collect_schema().names():
            raise ValueError(f'df_fai must have a "len" column')

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
            #print(f'{depth} - {index}')
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

            #print(f'\t* {", ".join([str(x) for x in index_col_list[-1]])}')

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
                    }, schema=df_depth.schema),
                    df_depth
                ])
                .select(df_depth.columns)
            )
        else:
            df_depth = df_depth.filter(pl.col('depth') > 0)

        df_depth_list.append(df_depth)

    # Concat
    df_depth = (
        pl.concat(df_depth_list)
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
    """
    Filter alignments based on overlap with deeply-mapped regions.

    Intersect df with df_depth and count the number of bases in df overlapping all records in df_depth (may overlap
    multiple records, intersect bases are summed). If the proportion of intersected bases exceeds max_overlap, append
    "append_filter" to the "filter" column in df. A reasonable max_overlap threshold will permit long alignments to
    pass through deeply-mapped regions without being filtered.

    Args:
        df: Table of alignment records.
        df_depth: Table of depth records. If None, will be generated from df.
        max_depth: Maximum depth, filter records intersecting loci exceeding this depth (> max_depth).
        max_overlap: Maximum overlap allowed when intersecting deep alignment regions.
        coord_cols: Column names for coordinates (chrom, start, end).
        append_filter: String to append to "filter" column in df.

    Returns:
        df with "filter" column appended.

    Raises:
        ValueError: Arguments are invalid.
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
    """
    Intersect tables by coordinates and count the number bases in df overlapping all records in df_other.

    Warning:
        If df_other contains overlapping records, the intersect bases will be counted multiple times. This function
        is typically used with a depth table, which does not contain overlaps.

    Args:
        df: Table with coordinates.
        df_other: Other table with coordinates.
        coord_cols: Coordinate columns to use for depth. Typically ('chrom', 'pos', 'end') or
            ('qry_id', 'qry_pos', 'qry_end'). If a string, must be "ref" (for chrom, pos, end) or
            "qry" (for qry_id, qry_pos, qry_end).

    Returns:
        A table with the following fields:
            index: Index in df by position (first record is 0, secord is 1, etc).
            len: Length of the record (end position - start position).
            bp: Number of bases in df overlapping all records in df_other.
            bp_prop: Proportion of bases in df overlapping all records in df_other.

    Raises:
        ValueError: Arguments are invalid.
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
    """
    Checks coord_cols and substitutes defaults.

    Convenience method for checking coord_cols parameters used by several functions.

    Args:
        coord_cols: A tuple of three elements are a pre-defined keyword indicating which coordinate columns to use.

    Returns:
        A a tuple af three column names, e.g. ('chrom', 'pos', 'end').
    """

    if coord_cols is None:
        raise ValueError('coord_cols must be specified')

    if isinstance(coord_cols, str):
        coord_cols: tuple[str, str, str] = NAMED_COORD_COLS.get(coord_cols, None)

        if coord_cols is None:
            raise ValueError(f'If coord_cols is a string, it must be "ref" or "qry"')

    coord_cols = tuple(coord_cols)

    if len(coord_cols) != 3:
        raise ValueError(
            f'coord_cols must have length 3 [(chrom, pos, end) or (qry_id, qry_pos, qry_end)), length={len(coord_cols)}'
        )

    return coord_cols


def get_qry_order(df):
    """
    Get a column describing the query order of each alignment record.

    For any query sequence, the first alignment record in the sequence (i.e. containing the left-most aligned base
    relative to the query sequence) will have order 0, the next alignment record 1, etc. The order is set per query
    sequence (i.e. the first aligned record of every unique query ID will have order 0).

    Args:
        df: DataFrame of alignment records.

    Returns:
        A Series of alignment record query orders.
    """

    return (
        df
        .with_row_index('index')
        .sort(['qry_id', 'qry_pos', 'qry_end'])
        .drop('qry_order', strict=False)
        .with_row_index('qry_order')
        .select(
            pl.col('index'),
            (pl.col('qry_order') - pl.col('qry_order').first().over('qry_id')).alias('qry_order')
        )
        .sort('index')
        .select(
            pl.col('qry_order')
        )
        .to_series()
    )


def align_stats(
        df: pl.DataFrame,
        head_cols: Optional[list[tuple[str, str]]] = None
) -> pl.DataFrame:
    """
    Collect high-level stats from an alignment table separated by pass (filter list is empty) and fail (filter list is
    not empty).

    Args:
        df: Alignment table.
        head_cols: List of tuples of (column name, literal value) to prepend to the output table.

    Returns:
        A table with new or updated feature columns.
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


# def aggregate_alignment_records(
#         df_align: pd.DataFrame,
#         df_qry_fai: pd.Series,
#         score_model: bool=None,
#         min_score: float=None,
#         noncolinear_penalty: bool=True
# ):
#     """
#     Aggregate colinear alignment records.
#
#     :param df_align: Table of alignment records. MUST be query trimmed (or query- & reference-trimmed)
#     :param df_qry_fai: Query FAI.
#     :param score_model: Model for scoring INS and DEL between alignment records. If none, use the default model.
#     :param min_score: Do not aggregate alignment records with a score below this value. Defaults to the score of a
#         10 kbp gap.
#     :param noncolinear_penalty: When aggregating two records, add a gap penalty equal to the difference between the
#         unaligned reference and query bases between the records. This penalizes non-colinear alignments.
#
#     :return: Table of aggregated alignment records.
#     """
#
#     # Check parameters
#     if score_model is None:
#         score_model = score.get_score_model()
#
#     if min_score is None:
#         min_score = score_model.gap(10000)
#
#     min_agg_index = int(10 ** np.ceil(np.log10(
#         np.max(df_align['INDEX'])
#     ))) - 1  # Start index for aggregated records at the next power of 10
#
#     next_agg_index = min_agg_index + 1
#
#     # Sort
#     df_align = df_align.sort_values(['QRY_ID', 'QRY_POS']).copy()
#     df_align['INDEX_PREAGG'] = df_align['INDEX']
#
#     # Return existing table if empty
#     if df_align.shape[0] == 0:
#         df_align['INDEX_PREAGG'] = df_align['INDEX']
#         return df_align
#
#     df_align['INDEX_PREAGG'] = df_align['INDEX'].apply(lambda val: [val])
#     df_align['MAPQ'] = df_align['MAPQ'].apply(lambda val: [val])
#     df_align['FLAGS'] = df_align['FLAGS'].apply(lambda val: [val])
#
#     # Find and aggregate near co-linear records over SVs
#     align_records = list()  # Records that were included in a merge
#
#     for qry_id in sorted(set(df_align['QRY_ID'])):
#         df = df_align.loc[df_align['QRY_ID'] == qry_id]
#         i_max = df.shape[0] - 1
#
#         i = 0
#         row1 = df.iloc[i]
#
#         while i < i_max:
#             i += 1
#
#             row2 = row1
#             row1 = df.iloc[i]
#
#             # Skip if chrom or orientation is not the same
#             if row1['#CHROM'] != row2['#CHROM'] or row1['IS_REV'] != row2['IS_REV']:
#                 align_records.append(row2)
#                 continue
#
#             # Get reference distance
#             if row1['IS_REV']:
#                 ref_dist = row2['POS'] - row1['END']
#             else:
#                 ref_dist = row1['POS'] - row2['END']
#
#             qry_dist = row1['QRY_POS'] - row2['QRY_END']
#
#             if qry_dist < 0:
#                 raise RuntimeError(f'Query distance is negative: {qry_dist}: alignment indexes {row1["INDEX"]} and {row2["INDEX"]}')
#
#             if ref_dist >= 0:
#                 # Contiguous in reference space, check query space
#
#                 # Score gap between the alignment records
#                 this_score = score_model.gap(ref_dist) + score_model.gap(qry_dist)
#
#                 score_gap = this_score + (
#                     score_model.gap(np.abs(qry_dist - ref_dist)) if noncolinear_penalty else 0
#                 )
#
#                 if score_gap < min_score:
#                     align_records.append(row2)
#                     continue
#
#                 #
#                 # Aggregate
#                 #
#                 row1 = row1.copy()
#
#                 # Set query position
#                 row1['QRY_POS'] = row2['QRY_POS']
#
#                 # Get rows in order
#                 if row1['IS_REV']:
#                     row_l = row1
#                     row_r = row2
#
#                     row1['END'] = row2['END']
#
#                     row1['TRIM_REF_R'] = row2['TRIM_REF_R']
#                     row1['TRIM_QRY_R'] = row2['TRIM_QRY_R']
#
#                 else:
#                     row_l = row2
#                     row_r = row1
#
#                     row1['POS'] = row2['POS']
#
#                     row1['TRIM_REF_L'] = row2['TRIM_REF_L']
#                     row1['TRIM_QRY_L'] = row2['TRIM_QRY_L']
#
#                 # Set records
#                 row1['FLAGS'] = row1['FLAGS'] + row2['FLAGS']
#                 row1['MAPQ'] = row1['MAPQ'] + row2['MAPQ']
#                 row1['INDEX_PREAGG'] = row1['INDEX_PREAGG'] + row2['INDEX_PREAGG']
#                 row1['SCORE'] = row1['SCORE'] + row2['SCORE']
#
#                 if 'RG' in row1:
#                     row1['RG'] = np.nan
#
#                 if 'AO' in row1:
#                     row1['AO'] = np.nan
#
#                 # Merge CIGAR strings
#                 op_arr_l = op.cigar_as_array(row_l['CIGAR'])
#                 op_arr_r = op.cigar_as_array(row_r['CIGAR'])
#
#                 while op_arr_l.shape[0] > 0 and op_arr_l[-1, 0] in op.CLIP_SET:  # Tail of left record
#                     op_arr_l = op_arr_l[:-1]
#
#                 while op_arr_r.shape[0] > 0 and op_arr_r[0, 0] in op.CLIP_SET:  # Head of right record
#                     op_arr_r = op_arr_r[1:]
#
#                 ins_len = qry_dist
#                 del_len = ref_dist
#
#                 if qry_dist > 0:
#                     while op_arr_l.shape[0] > 0 and op_arr_l[-1, 0] == op.I:  # Concat insertions (no "...xIxI..." in CIGAR)
#                         ins_len += op_arr_l[-1, 1]
#                         op_arr_l = op_arr_l[:-1]
#
#                     op_arr_l = np.append(op_arr_l, np.array([[ins_len, op.I]]), axis=0)
#
#                 if ref_dist > 0:
#                     while op_arr_l.shape[0] > 0 and op_arr_l[-1, 0] == op.D:  # Concat deletions (no "...xDxD..." in CIGAR)
#                         del_len += op_arr_l[-1, 1]
#                         op_arr_l = op_arr_l[:-1]
#
#                     op_arr_l = np.append(op_arr_l, np.array([[del_len, op.D]]), axis=0)
#
#                 op_arr = np.append(op_arr_l, op_arr_r, axis=0)
#
#                 row1['CIGAR'] = op.to_cigar_string(op_arr)
#                 row1['SCORE'] = score_model.score_operations(op_arr)
#
#                 # Set alignment indexes
#                 if row2['INDEX'] < min_agg_index:
#                     # Use the next aggregate index
#                     row1['INDEX'] = next_agg_index
#                     next_agg_index += 1
#
#                 else:
#                     # row2 was aggregated, use its aggregate index
#                     row1['INDEX'] = row2['INDEX']
#
#                 # Check
#                 records.check_record(row1, df_qry_fai)
#
#             else:
#                 align_records.append(row2)
#
#         # Add last record
#         align_records.append(row1)
#
#     # Concatenate records
#     df = pd.concat(align_records, axis=1).T
#
#     df['MAPQ'] = df['MAPQ'].apply(lambda val: ','.join([str(v) for v in val]))
#     df['FLAGS'] = df['FLAGS'].apply(lambda val: ','.join([str(v) for v in val]))
#     df['INDEX_PREAGG'] = df['INDEX_PREAGG'].apply(lambda val: ','.join([str(v) for v in val]))
#
#     # Assign order per query sequence
#     df['QRY_ORDER'] = get_qry_order(df)
#
#     # Reference order
#     df.sort_values(['#CHROM', 'POS', 'END', 'QRY_ID'], ascending=[True, True, False, True], inplace=True)
#
#     # Check sanity (whole table, modified records already checked, should pass)
#     df.apply(records.check_record, df_qry_fai=df_qry_fai, axis=1)
#
#     return df
