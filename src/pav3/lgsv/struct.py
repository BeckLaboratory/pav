"""Generate structure annotations for inter-alignment variants."""

__all__ = [
    'get_ref_trace',
    'smooth_ref_trace',
    'qry_trace_str',
    'ref_trace_str',
]

import math
from typing import Optional

import polars as pl

from ..align.tables import align_depth_table
from ..const import DEFAULT_LG_SMOOTH_SEGMENTS

from .interval import AnchoredInterval

REF_TRACE_SCHEMA = {
    'chrom': pl.String,
    'pos': pl.Int32,
    'end': pl.Int32,
    'depth': pl.Int32,
    'fwd_count': pl.Int32,
    'rev_count': pl.Int32,
    'seg_index': pl.List(pl.Int32),
    'type': pl.String,
}
"""Schema for reference trace tables."""


def get_ref_trace(
        interval: AnchoredInterval,
        df_ref_fai: pl.DataFrame = None,
        smooth_factor: float = 0.0,
        varlen: Optional[int] = None,
        is_pass: Optional[bool] = None
):
    """Get a table representing the trace across the reference locus for this SV.

    This only covers the locus of the SV and omits distal template switches.

    In some cases, a complex SV has no reference context. If there is a clean insertion site (no deleted or duplicated
    bases at the SV site), but the SV is templated from multiple locations (or contains templated and untemplated
    insertions). In these cases, an empty table is returned.

    :param interval: Interval to generate a reference trace for.
    :param df_ref_fai: Reference FASTA index table.
    :param smooth_factor: Smoothing factor.
    :param varlen: Variant length. Required for smoothing. May be `None` if smooth_factor is zero.
    :param is_pass: Variant passed filters. If True, poor alignment segments are removed from the variant call,
        otherwise, they are retained and the full erroneous CSV structure is reported as a filtered variant. If `None`,
        the variant is assumed to pass if both anchors are not filtered.

    :returns: A reference trace table. Has no rows if there is no reference context at the SV site.
    """
    df_segment = interval.df_segment

    if is_pass is None:
        is_pass = interval.all_anchor_pass

    if interval.n_anchor != 2:
        raise ValueError(f'Interval must have exactly two anchors, got {interval.n_anchor}')

    if is_pass:
        df_segment = df_segment.filter(pl.col('is_anchor') | (pl.col('filter_pass') & pl.col('is_aligned')))

    # Set local template switch boundaries
    # Distance from reference positions (pos & end) must be no more half of:
    #   * CPX event in query bases.
    #   * Distance between reference anchoring breakpoints.
    # May not expand beyond anchoring alignments.
    local_dist = max([len(interval.region_ref), len(interval.region_qry)]) // 2

    local_pos = max([
        interval.region_ref.pos - local_dist,
        min([
            df_segment[0, 'pos'],
            df_segment[0, 'end'],
            df_segment[-1, 'pos'],
            df_segment[-1, 'end']
        ])
    ])

    local_end = min([
        interval.region_ref.end + local_dist,
        max([
            df_segment[0, 'pos'],
            df_segment[0, 'end'],
            df_segment[-1, 'pos'],
            df_segment[-1, 'end']
        ])
    ])

    # Get reference trace table
    df_ref_trace = (
        align_depth_table(
            df_segment.filter('is_aligned'),
            df_fai=df_ref_fai,
            retain_filtered=not interval.all_anchor_pass
        )
        .lazy()

        # Remove distal regions
        .with_columns(
            pl.col('pos').clip(lower_bound=local_pos),
            pl.col('end').clip(upper_bound=local_end)
        )
        .filter(
            (pl.col('chrom') == interval.chrom) & (pl.col('end') - pl.col('pos') > 0)
        )

        # Rename  & drop
        .rename({'index': '_index_aligned'})
        .drop('depth')
    )

    df_ref_trace = (
        df_ref_trace

        # Add annotations from the segment table
        .with_row_index('_index_trace')
        .join(
            (
                df_ref_trace
                .with_row_index('_index_trace')
                .explode('_index_aligned')
                .join(
                    (
                        (
                            df_segment
                            .lazy()
                            .with_columns(
                                pl.when(pl.col('is_aligned'))
                                .then(pl.col('is_aligned').cum_sum().over(pl.col('is_aligned')) - 1)
                                .otherwise(None)
                                .alias('_index_aligned')
                            )
                        )
                        .select(['_index_aligned', 'is_rev', 'is_anchor', 'seg_index'])
                    ),
                    on='_index_aligned',
                    how='left'
                )
                .group_by('_index_trace')
                .agg(
                    (
                        pl.col('is_rev').drop_nulls().sum() +
                        (~ pl.col('is_rev').drop_nulls()).sum()
                    ).alias('depth'),
                    (~ pl.col('is_rev').drop_nulls()).sum().alias('fwd_count'),
                    pl.col('is_rev').drop_nulls().sum().alias('rev_count'),
                    (pl.col('is_anchor').sum() == pl.len()).alias('_all_anchor'),
                    pl.col('seg_index').drop_nulls().explode().alias('seg_index'),
                )
            ),
            on='_index_trace'
        )
        .drop(['_index_aligned', '_index_trace'])

        # Drop anchor-only segments at the edges
        .filter(
            ~pl.col('_all_anchor') | (
                (pl.col('_all_anchor').rle_id() > 0) & (pl.col('_all_anchor').reverse().rle_id().reverse() > 0)
            )
        )
        .drop('_all_anchor')

        # Annotate segment types
        .with_columns(
            (
                pl.when(pl.col('depth') == 0)
                .then(pl.lit('DEL'))
                .when((pl.col('fwd_count') == 1) & (pl.col('rev_count') == 0))
                .then(pl.lit('NML'))
                .when((pl.col('fwd_count') == 0) & (pl.col('rev_count') == 1))
                .then(pl.lit('INV'))
                .when(pl.col('depth') == 2)
                .then(pl.lit('DUP'))
                .when(pl.col('depth') == 3)
                .then(pl.lit('TRP'))
                .when(pl.col('depth') == 4)
                .then(pl.lit('QUAD'))
                .otherwise(pl.lit('HDUP'))
            ).alias('type')
        )
        .with_columns(
            pl.when((pl.col('depth') < 2) | (pl.col('rev_count') == 0))
            .then(pl.col('type'))
            .when(pl.col('fwd_count') <= 1)
            .then(pl.concat_str(pl.lit('INV'), pl.col('type')))
            .when(pl.col('fwd_count') > 1)
            .then(pl.concat_str(pl.lit('MIX'), pl.col('type')))
            .alias('type')
        )

        # Complete table
        .collect()
    )

    if smooth_factor > 0.0:
        df_ref_trace = smooth_ref_trace(df_ref_trace, varlen, smooth_factor)

    # if list(df_ref_trace.columns) != REF_TRACE_COLUMNS:
    #     raise RuntimeError(f'Unexpected reference trace columns (Program bug): {", ".join(df_ref_trace.columns)}')

    return (
        df_ref_trace
        .cast(REF_TRACE_SCHEMA)
        .select(REF_TRACE_SCHEMA.keys())
    )


def smooth_ref_trace(
        df_ref_trace: pl.DataFrame,
        varlen: int,
        smooth_factor: float = DEFAULT_LG_SMOOTH_SEGMENTS,
):
    """Smooth a reference trace table.

    :param df_ref_trace: Reference trace table.
    :param smooth_factor: Smoothing factor.
    :param varlen: Variant length.

    :returns: Smoothed reference trace table.
    """
    if smooth_factor <= 0.0:
        return df_ref_trace

    if varlen is None or varlen == 0:
        raise RuntimeError(f'Variant length (varlen) is required for smoothing reference trace: {varlen}')

    min_len = math.ceil(varlen * smooth_factor)

    df_ref_trace = (
        df_ref_trace

        # Mark diminutive segments (too small to keep)
        .drop('_is_dim', strict=False)
        .with_columns(
            (pl.col('end') - pl.col('pos') < min_len).alias('_is_dim')  # Is a diminutive segment
        )

        # Drop diminutive segments at edges (top or bottom of the trace)
        .filter(
            ~pl.col('_is_dim') | pl.col('_is_dim').rle_id() > 0,
            ~pl.col('_is_dim') | pl.col('_is_dim').reverse().rle_id().reverse() > 0
        )
    )

    # No trace if all segments are diminutive or table is empty
    if df_ref_trace.select((~ pl.col('_is_dim')).sum()).item() == 0:
        return df_ref_trace.filter(~ pl.col('_is_dim'))

    # Search for diminutive segments
    # TODO: Streamline this section, it's ugly
    df_trace_list = list()

    last_index = df_ref_trace.height

    start_index = 0
    start_row = df_ref_trace.row(start_index, named=True)
    seg_index_list = []

    while start_row is not None:

        # Next end
        end_index = start_index + 1

        # End of trace
        if end_index == last_index:
            df_trace_list.append(start_row)
            start_row = None
            continue

        # Find end of this collapse
        end_row = None

        skip_len = 0
        is_compat = False

        while end_index < last_index and end_row is None:
            next_end_row = df_ref_trace.row(end_index, named=True)

            if next_end_row['chrom'] != start_row['chrom']:
                break

            if not next_end_row['_is_dim']:
                end_row = next_end_row
                break

            if (
                    start_row['fwd_count'] == next_end_row['fwd_count'] and
                    start_row['rev_count'] == next_end_row['rev_count']
            ):
                end_row = next_end_row
                is_compat = True
                break

            skip_len += next_end_row['end'] - next_end_row['pos']
            seg_index_list += [i for i in next_end_row['seg_index'] if i not in seg_index_list]
            end_index += 1

        if end_index == last_index:
            df_trace_list.append(start_row)

            start_index += 1
            while start_index < last_index:
                df_trace_list.append(df_ref_trace.row(start_index, named=True))
                start_index += 1

            break

        end_row = df_ref_trace.row(end_index, named=True)

        if is_compat:
            # Concat range
            start_row['end'] = end_row['end']
            start_row['seg_index'] = seg_index_list + [i for i in end_row['seg_index'] if i not in seg_index_list]

            start_index = end_index

        elif end_index > start_index + 1:
            # Split range
            skip_len_l = skip_len // 2
            skip_len_r = skip_len - skip_len_l

            start_row['end'] = start_row['end'] + skip_len_l
            start_row['seg_index'] += [i for i in seg_index_list if i not in start_row['seg_index']]

            end_row['pos'] = end_row['pos'] - skip_len_r
            end_row['seg_index'] += [i for i in seg_index_list if i not in end_row['seg_index']]

            df_trace_list.append(start_row)

            start_row = end_row
            start_index = end_index

        else:
            df_trace_list.append(start_row)
            start_index = end_index

            start_row = df_ref_trace.row(start_index, named=True) if start_index < last_index else None

    df_ref_trace = pl.DataFrame(df_trace_list, orient='row').drop('_is_dim')

    # Concatenate adjacent segments with the same counts
    return (
        df_ref_trace
        .with_columns(
            pl.concat_list(['fwd_count', 'rev_count']).rle_id().alias('_count_rle')
        )
        .group_by('_count_rle')
        .agg(
            pl.col('pos').min(),
            pl.col('end').max(),
            pl.col('seg_index').explode().unique().alias('seg_index'),
            *[
                pl.col(col).first() for col in df_ref_trace.columns if col not in [
                    'pos', 'end', 'seg_index'
                ]
            ]
        )
        .select(df_ref_trace.columns)
    )


def qry_trace_str(
        df_segment: pl.DataFrame,
        is_pass: Optional[bool] = None,
) -> str:
    """Get a string representing the query path through template switches, duplicated regions, and insertions.

    :param df_segment: Segment table.
    :param is_pass: Variant passed filters. If True, poor alignment segments are removed from the variant call,
        otherwise, they are retained and the full erroneous CSV structure is reported as a filtered variant. If
        `None`, the variant is assumed to pass if both anchors are not filtered.

    :returns: Query trace string.
    """
    if df_segment.height < 2 or not df_segment[0, 'is_anchor'] or not df_segment[-1, 'is_anchor']:
        raise ValueError('Invalid segment table: Table should begin and end with anchor segments')

    if is_pass is None:
        is_pass = df_segment.filter(pl.col('is_anchor')).select(pl.col('filter_pass').all()).item()

    if is_pass:
        df_segment = df_segment.filter(pl.col('is_anchor') | (pl.col('filter_pass') & pl.col('is_aligned')))

    df_segment = (
        df_segment
        .sort(['qry_id', 'qry_pos', 'qry_end'])
        .drop('strand', strict=False)
    )

    if df_segment.filter(pl.col('is_anchor')).select(pl.col('is_rev').all()).item():
        df_segment = (
            df_segment
            .with_columns(
                pl.col('qry_end').alias('qry_pos'),
                pl.col('qry_pos').alias('qry_end'),
                ~ pl.col('is_rev'),
            )
            .reverse()
        )

    df_segment = df_segment.with_columns(
        pl.col('is_rev').cast(pl.String).replace(
            ['true', 'false'],
            ['-', '+']
        )
        .fill_null('.')
        .alias('strand')
    )

    # Make structure table
    row = df_segment.row(0, named=True)
    last_chrom = row['chrom']
    last_pos = row['end']

    struct_list = []

    for row in df_segment.filter(~ pl.col('is_anchor')).iter_rows(named=True):
        if row['is_aligned']:

            if row['chrom'] != last_chrom:
                last_chrom = row['chrom']
                struct_list.append(f'TSCHR[{last_chrom}-{row["pos"]:,}({row["strand"]})]')
            else:
                struct_list.append(f'TS[{row["pos"] - last_pos:,}({row["strand"]})]')

            struct_list.append(f'DUP[{row["len_ref"]:,}-{row["len_qry"]:,}({row["strand"]})]')

        else:
            struct_list.append(f'INS[{row["len_qry"]:,}({row["strand"]})]')

    row = df_segment.row(-1, named=True)

    if row['chrom'] != last_chrom:
        last_chrom = row['chrom']
        struct_list.append(f'TSCHR[{last_chrom}:{row["pos"]:,}({row["strand"]})]')
    else:
        struct_list.append(f'TS[{row["pos"] - last_pos:,}({row["strand"]})]')

    return ':'.join(struct_list)


def ref_trace_str(
        df_ref_trace: pl.DataFrame,
        with_len: bool = True,
):
    """Get reference structure string describing a complex SV from the reference perspective.

    :param df_ref_trace: Reference trace table.
    :param with_len: Whether to include the length of each segment in the structure string.

    :returns: A string describing the reference structure.
    """
    if with_len:
        fmt_expr = pl.format(
            '{}[{}]',
            pl.col('type'),
            (pl.col('end') - pl.col('pos')).map_elements(lambda x: f'{x:,}', return_dtype=pl.String)
        )
    else:
        fmt_expr = pl.col('type')

    return (
        df_ref_trace
        .select(fmt_expr.str.join(':'))
        .item()
    )
