"""Routines for creating tracks."""

__all__ = [
    'TRACK_RESOURCE',
    'align',
]

from collections.abc import (
    Iterable,
    Iterator,
)
import itertools
import importlib.resources
from pathlib import Path
import re
from typing import Optional

import polars as pl

from . import const as fig_const
from . import util as fig_util
from ..call import expr

TRACK_RESOURCE = 'pav3.data.tracks'

ALIGN_TRACK_TABLE_FILENAME = 'alignment_track_fields.tsv'

VAR_TRACK_TABLE_FILENAME = 'variant_track_fields.tsv'

TRIM_DESC = {
    'none': 'None',
    'qry': 'Qry',
    'qryref': 'Qry-Ref',
}


#
# Variant tracks
#


def call_hap_insdel(
        callset_def: dict[str, str | Path],
        df_fai: pl.LazyFrame | pl.DataFrame,
        field_path: Optional[Path | str] = None,
        thick_lines: bool = False,
        center_ins: bool = True,
) -> tuple[pl.LazyFrame, list[str]]:
    """Create a track of variant calls.

    :param callset_def: Callset definition.
    :param df_fai: Reference table with "chrom" and "len" columns.
    :param field_path: Path to field table describing track fields.
    :param thick_lines: If True, draw variants with thick lines. Otherwise, use thick for insertion site.
    :param center_ins: If True, center insertions on the variant position. Otherwise, use the start position.

    :return: Tuple of variant LazyFrame and AS file lines.
    """
    mpl_local = fig_util.require_matplotlib()

    if isinstance(df_fai, pl.DataFrame):
        df_fai = df_fai.lazy()

    asm_name = asm_name if (asm_name := callset_def.get('asm_name', '').strip()) else 'Unknown'
    hap = hap if (hap := callset_def.get('hap', '').strip()) else 'Unknown'
    vartype = (vartype if (vartype := callset_def.get('varclass', '').strip()) else 'unknown').lower()
    varclass = (varclass if (varclass := callset_def.get('varclass', '').strip()) else 'svindel').lower()

    track_desc_short = f'PAVCallHap{asm_name}{hap}{vartype.capitalize()}'
    track_desc_long = f'PAV Call Hap ({asm_name}-{hap}: {vartype.capitalize()})'

    # Read field table
    if field_path is None:
        if not importlib.resources.is_resource(TRACK_RESOURCE, VAR_TRACK_TABLE_FILENAME):
            raise FileNotFoundError(f'Track field table not found in resource: {TRACK_RESOURCE}.{VAR_TRACK_TABLE_FILENAME}')

        field_path = importlib.resources.files(TRACK_RESOURCE).joinpath(VAR_TRACK_TABLE_FILENAME)
    else:
        field_path = Path(field_path)

    if not field_path.is_file():
        raise FileNotFoundError(f'Field table not found: {field_path}')

    df_as = pl.read_csv(field_path, separator='\t')

    # Color table
    color_table = pl.DataFrame(
        [
            {
                'vartype': vartype,
                '_is_pass': is_pass,
                'color': (
                    fig_util.color_to_ucsc_string(
                        color if is_pass else fig_util.lighten_color(color, 0.25)
                    )
                )
            }
            for vartype, color in fig_const.COLOR_VARTYPE.items()
            for is_pass in (True, False)
        ],
        orient='row',
        schema={
            'vartype': pl.String,
            '_is_pass': pl.Boolean,
            'color': pl.String,
        },
    ).lazy()

    # Read callset
    callset_path = Path(p) if (p := callset_def.get('callset_path', None)) is not None else None

    if callset_path is None or not callset_path.is_file():
        raise FileNotFoundError(f'Callset file not found: {callset_path}')

    # Format table
    filter_list = []

    if varclass == 'sv':
        filter_list.append(pl.col('varlen') >= 50)
    elif varclass == 'indel':
        filter_list.append(pl.col('varlen') >= 50)
    elif varclass != 'svindel':
        raise ValueError('Unknown variant class: {varclass}')

    df = (
        pl.scan_parquet(callset_path)
        .with_row_index('_index')
        .with_columns((pl.col('filter').list.len() == 0).alias('_is_pass'))
        .join(color_table, on=['vartype', '_is_pass'], how='left')
        .join(
            df_fai.select(
                'chrom', pl.col('len').alias('_chrom_len')
            ),
            on='chrom', how='left',
        )
        .with_columns(
            pl.col('color').fill_null(fig_util.color_to_ucsc_string((0, 0, 0))),
            pl.col('filter').list.join(',').cast(pl.String).alias('filter'),
            pl.lit(asm_name).alias('asm_name'),
            pl.lit(hap).alias('hap'),
            expr.qry_region().alias('qry_region'),
            (pl.when('qry_rev').then(pl.lit('-')).otherwise(pl.lit('+'))).cast(pl.String).alias('strand'),
            pl.col('align_index').cast(pl.List(pl.String)).list.join(',').cast(pl.String).alias('align_index'),
            pl.col('call_source').fill_null('.'),
            pl.col('varsubtype').fill_null('.'),
            (
                pl.when(pl.col('hom_ref').is_not_null())
                .then(pl.concat_str(
                    pl.col('hom_ref').struct.field('up'),
                    pl.lit(','),
                    pl.col('hom_ref').struct.field('dn'),
                ))
                .otherwise(pl.lit('.'))
                .alias('hom_ref')
            ),
            (
                pl.when(pl.col('hom_qry').is_not_null())
                .then(pl.concat_str(
                    pl.col('hom_qry').struct.field('up'),
                    pl.lit(','),
                    pl.col('hom_qry').struct.field('dn'),
                ))
                .otherwise(pl.lit('.'))
                .alias('hom_qry')
            ),
            pl.col('discord').list.join(','),
            pl.col('inner').list.join(','),
            pl.col('dup').list.eval(
                pl.concat_str(
                    pl.element().struct.field('chrom'),
                    pl.lit(':'),
                    pl.element().struct.field('pos'),
                    pl.lit('-'),
                    pl.element().struct.field('end'),
                    pl.lit('('),
                    (
                        pl.when(pl.element().struct.field('is_rev'))
                        .then(pl.lit('-'))
                        .otherwise(pl.lit('+'))
                    ),
                    pl.lit(')'),
                )
            ).list.join(','),
            pl.lit(1000).alias('track_score'),
            (
                pl.when(pl.col('seq').str.len_chars() > 20)
                .then(
                    pl.concat_str(
                        pl.col('seq').str.slice(0, 10),
                        pl.lit('...('),
                        pl.col('seq').str.len_chars() - 20,
                        pl.lit(')...'),
                        pl.col('seq').str.slice(-10, 10),
                    )
                )
                .otherwise('seq')
            )
        )
    )

    if thick_lines:
        df = (
            df
            .with_columns(
                (
                    pl.when(pl.col('vartype') == 'INS')
                    .then(pl.col('pos') + pl.col('varlen'))
                    .otherwise(pl.col('end'))
                )
                .clip(0, pl.col('_chrom_len'))
                .alias('end'),
            )
            .with_columns(
                pl.min_horizontal('pos', 'end').alias('pos_thick'),
                pl.max_horizontal('pos', 'end').alias('end_thick'),
            )
        )
    else:
        divider = 2 if center_ins else 1

        df = (
            df
            .with_columns(
                pl.col('pos').alias('pos_thick'),
                (
                    pl.when(pl.col('vartype') == 'INS')
                    .then('end')
                    .otherwise('pos')  # No thick part of pos_thick == end_thick
                    .alias('end_thick')
                ),
                (
                    pl.when(pl.col('vartype') == 'INS')
                    .then(
                        (pl.col('pos') + pl.col('varlen') // divider)
                        .clip(0, pl.col('_chrom_len'))
                    )
                    .otherwise('end')
                    .alias('end')
                )
            )
            .with_columns(
                pl.when(pl.col('vartype') == 'INS')
                .then(
                    # Shift back if centered or runs off chrom
                    (pl.col('end') - pl.col('varlen')).clip(0)
                )
                .otherwise('pos')
                .alias('pos')
            )
        )

    df = (
        df
        .sort('chrom', 'pos', 'end', '_index')
        .drop(
            '_is_pass', '_chrom_len', '_index',
            'qry_id', 'qry_pos', 'qry_end', 'qry_rev',
            strict=False,
        )
    )

    # Check columns
    col_set = set(df.collect_schema().names())

    if (missing_cols := set(df_as.filter('required').select('field').to_series()) - col_set):
        n = len(missing_cols)
        missing_cols = ', '.join(sorted(missing_cols)[:3]) + ('...' if n > 0 else '')

        raise ValueError(f'Missing {n} columns in alignment track definition {i}: {missing_cols}')

    df_as = df_as.filter(
        pl.col('field').is_in(col_set)
    )

    df = df.select(df_as.select('field').to_series().to_list())

    # AS file lines
    as_lines = [
        f'table {track_desc_short}',
        f'"{track_desc_long}"',
        f'('
    ] + [
        '{type} {name}; "{desc}"'.format(**row)
        for row in df_as.iter_rows(named=True)
    ] + [
        ')'
    ]

    return df, as_lines




#
# Alignment tracks
#

def align(
        align_defs: list[dict[str, str | Path]],
        field_path: Optional[Path | str] = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Create alignment track.

    :param trim: Trim type (none, qry, qryref).
    :param field_path: Path to field table describing track fields.

    :return: Tuple of alignment DataFrame and AS file lines.
    """
    mpl_local = fig_util.require_matplotlib()

    # Get track description
    track_desc_short, track_desc_long = get_track_desc_align(
        'PAV Align',
        {
            re.sub(r'[^a-zA-Z0-9_]', '', str(d.get('trim', None)).strip().lower())
            for d in align_defs if d.get('trim', None) is not None
        }
    )

    # Read field table
    if field_path is None:
        if not importlib.resources.is_resource(TRACK_RESOURCE, ALIGN_TRACK_TABLE_FILENAME):
            raise FileNotFoundError(f'Track field table not found in resource: {TRACK_RESOURCE}.{ALIGN_TRACK_TABLE_FILENAME}')

        field_path = importlib.resources.files(TRACK_RESOURCE).joinpath(ALIGN_TRACK_TABLE_FILENAME)
    else:
        field_path = Path(field_path)

    if not field_path.is_file():
        raise FileNotFoundError(f'Field table not found: {field_path}')

    df_as = pl.read_csv(field_path, separator='\t')

    # Initialise colors
    colormap = mpl_local.colormaps[fig_const.ALIGN_COLORMAP]
    rotate_iter = rotate_map()

    # Read alignments
    df_list = list()

    for i, align_def in enumerate(align_defs):
        align_path = Path(p) if (p := align_def.get('align_path', None)) is not None else None
        hap = align_def.get('hap', None)
        asm_name = align_def.get('asm_name', '')
        trim = TRIM_DESC.get(align_def.get('trim', None), 'Unknown')

        color = colormap(next(rotate_iter))

        color_pass = pl.lit(
            fig_util.color_to_ucsc_string(color)
        ).cast(pl.String)

        color_fail = pl.lit(
            fig_util.color_to_ucsc_string(
                fig_util.lighten_color(color, 0.25)
            )
        ).cast(pl.String)

        if hap is None or not (hap := str(hap).strip()):
            raise ValueError(f'Haplotype not found for alignment definition index {i}')

        if align_path is None or not align_path.is_file():
            raise FileNotFoundError(f'Alignment file not found for alignment definition index {i}: {str(align_path)}')

        df_list.append(
            pl.scan_parquet(align_path)
            .with_row_index('_index')
            .drop('asm_name', 'hap', 'trim', 'align_ops', strict=False)
            .with_columns(
                pl.lit(asm_name).alias('asm_name'),
                pl.lit(hap).alias('hap'),
                pl.lit(trim).alias('trim'),
                (
                    pl.when(pl.col('filter').list.len() == 0)
                    .then(color_pass)
                    .otherwise(color_fail)
                ).alias('color'),
                pl.lit(i).alias('_align_def_index'),
                expr.qry_region().alias('qry_region'),
            )
        )

    df = (
        pl.concat(df_list)
        .sort('chrom', 'pos', 'end', '_align_def_index', '_index')

        .with_columns(
            pl.col('pos').alias('pos_thick'),
            pl.col('end').alias('end_thick'),
            pl.col('filter').list.join(',').cast(pl.String).alias('filter'),
            pl.concat_str(
                'qry_id',
                pl.lit(', hap='),
                'hap',
                pl.lit(', idx='),
                'align_index',
                pl.lit(', qry_ord='),
                'qry_order',
            ).alias('id'),
            (pl.col('match_prop') * 1000).clip(0.0, 1000.0).cast(pl.Int32).alias('track_score'),
            pl.col('rg').fill_null('').cast(pl.String).alias('rg'),
            (pl.when('is_rev').then(pl.lit('-')).otherwise(pl.lit('+'))).cast(pl.String).alias('strand'),
        )
        .drop('_align_def_index', '_index', 'qry_id', 'qry_pos', 'qry_end', 'is_rev')
        .collect()
    )

    # Check columns
    col_set = set(df.columns)

    if (missing_cols := set(df_as.filter('required').select('field').to_series()) - col_set):
        n = len(missing_cols)
        missing_cols = ', '.join(sorted(missing_cols)[:3]) + ('...' if n > 0 else '')

        raise ValueError(f'Missing {n} columns in variant track table: {missing_cols}')

    df_as = df_as.filter(
        pl.col('field').is_in(df.columns)
    )

    df = df.select(df_as.select('field').to_series().to_list())

    # AS file lines
    as_lines = [
        f'table {track_desc_short}',
        f'"{track_desc_long}"',
        f'('
    ] + [
        '{type} {name}; "{desc}"'.format(**row)
        for row in df_as.iter_rows(named=True)
    ] + [
        ')'
    ]

    return df, as_lines


def get_track_desc_align(
        track_desc: str = 'PAV Align',
        trim: Iterable[str] | str = (),
) -> tuple[str, str]:
    """Get track description with trim levels.

    :param track_desc: Track description.
    :param trim: Trim levels.

    :return: Tuple of short and long track descriptions.
    """

    trim_list = list(dict.fromkeys(
        [
            t2 for t2 in {
                str(t).strip() for t in
                ((trim,) if isinstance(trim, str) else trim)
                if t is not None
            } if t2
        ]
    ).keys())

    trim_list = [  # Each element is a tuple of the identifier and a description
        (t, TRIM_DESC.get(t.lower(), 'Unknown'))
        for t in trim_list
    ]

    track_desc_list = [
        t for t in re.split(r'[^A-Za-z_]+', track_desc) if t
    ]

    # Get track description
    if len(trim_list) == 1:
        trim, trim_desc = trim_list[0]
        trim_desc_short = trim.capitalize()

    elif len(trim) > 1:
        trim_desc_short = 'Multi'
        trim_desc = '/'.join(t[1] for t in trim_list)

    else:
        trim_desc_short = 'Unknown'
        trim_desc = 'Unknown'

    track_desc_short = f'{"".join([t.capitalize() for t in track_desc_list])}{trim_desc_short}'
    track_desc_long = f'{" ".join(track_desc_list)} (Trim {trim_desc})'

    return track_desc_short, track_desc_long


#
# Track utilities
#

def rotate_map() -> Iterator[float]:
    """Generate intervals in the range 0.0-1.0.

    This function aims to accomplish two goals:
    1) Iterate through a colormap in a way such that haplotype "n" always has the same color
        independent of the number of haplotypes in the callset.
    2) Separate colors as much as possible.

    The naive solution is to iterate through the colormap linearly in steps of 1/n, but this does
    not accomplish the second goal possibly coloring haplotypes in unexpected ways. For example,
    if a sample has a third "unphased" haplotype, then all colors for that sample would be
    different from samples with two haplotypes making it more difficult to visualize.

    :return: Infinite iterator of intervals in the range 0.0-1.0.
    """
    next_offset_iter = itertools.cycle([4, 8, 16, 32, 64])

    width = next(next_offset_iter)
    step = 1 / width

    i = 0.0

    while True:
        if i >= 1.0:
            width = next(next_offset_iter)
            step = 1 / width
            i = step / 2

        yield i
        i += step
