"""Integrate variant calls across variant call sources for single haplotypes.

Each function operates on a single haplotype, merging across haplotypes is not performed by this module.
"""

from typing import Optional

import polars as pl

from .expr import id_snv, id_nonsnv
from .. import schema

def read_trim_table(
    df_align_none: pl.LazyFrame,
    df_align_qry: pl.LazyFrame,
    df_align_qryref: pl.LazyFrame
) -> pl.DataFrame:
    """Get a table describing trimmed regions.

    :param df_align_none: Alignment table with no trimming.
    :param df_align_qry: Alignment table after query trimming.
    :param df_align_qryref: Alignment table after query and reference trimming.
    """
    df_trim_region_list = []

    for df_org, df_trim, filter_str in (
        (df_align_none, df_align_qry, 'TRIMQRY'),
        (df_align_qry, df_align_qryref, 'TRIMREF'),
    ):

        df = (
            df_org
            .select(['align_index', 'qry_id', 'qry_pos', 'qry_end'])
            .join(
                (
                    df_trim
                    .select(['align_index', 'qry_id', 'qry_pos', 'qry_end'])
                ),
                on='align_index',
                how='left',
                suffix='_trim'
            )
        )

        df_trim_region_list.append(
            pl.concat(
                [
                    (  # Left trim
                        df
                        .filter(pl.col('qry_pos_trim').is_not_null())
                        .select(
                            pl.col('align_index'),
                            pl.col('qry_id'),
                            pl.col('qry_pos'),
                            pl.col('qry_pos_trim').alias('qry_end'),
                            pl.lit(filter_str).alias('filter')
                        )
                    ),
                    (  # Right trim
                        df
                        .filter(pl.col('qry_pos_trim').is_not_null())
                        .select(
                            pl.col('align_index'),
                            pl.col('qry_id'),
                            pl.col('qry_end_trim').alias('qry_pos'),
                            pl.col('qry_end'),
                            pl.lit(filter_str).alias('filter')
                        )
                    ),
                    (  # Whole record trimmed
                        df
                        .filter(
                            pl.col('qry_pos_trim').is_null()
                        )
                        .select(
                            pl.col('align_index'),
                            pl.col('qry_id'),
                            pl.col('qry_pos'),
                            pl.col('qry_end'),
                            pl.lit(filter_str).alias('filter')
                        )
                    )
                ]
            )
        )

    return (
        pl.concat(df_trim_region_list)
        .sort(['qry_id', 'qry_pos', 'qry_end'])
        .collect()
    )


def id_and_version(
        df: pl.LazyFrame | pl.DataFrame,
        is_snv: bool = False,
        existing_ids: Optional[list[pl.LazyFrame | pl.DataFrame]] = None
) -> pl.LazyFrame:
    """Add and version IDs.

    Add or update the "id" column filling in all missing IDs and versioning them.

    :param df: DataFrame to add IDs to.
    :param is_snv: If True, use SNV ID generation.
    :param existing_ids: List of tables with existing IDs to use for versioning.

    :return: Table with complete IDs.
    """
    # Check columns on variant table
    col_list = df.collect_schema().names()

    if 'id' not in df.collect_schema().names():
        df = df.with_columns(pl.lit(None).alias('id'))
        col_list.append('id')

    if 'filter' not in df.collect_schema().names():
        df = df.with_columns(pl.lit([]).alias('filter'))

    # Fill missing IDs
    df = (
        df.with_columns(id_nonsnv())
    ) if not is_snv else (
        df.with_columns(id_snv())
    )

    # Get base ID and version
    df = (
        df
        .with_columns(
            (
                pl.col('id').str.extract_groups(r'([^.]*)(\.([0-9]+))?$')
                .struct.rename_fields(['_id_base', '_', '_id_version'])
                .struct.field('_id_base')
            )
        )
        .with_columns(
            (
                pl.col('filter').list.len()
                .rank(method='ordinal')
                .over(pl.col('_id_base'))
                - 1
            ).alias('_id_version')
        )
    )

    # Get maximum version for each existing ID
    if existing_ids is None:
        existing_ids = []

    existing_ids = [  # To lazy
        (df_existing.lazy() if isinstance(df_existing, pl.DataFrame) else df_existing)
            for df_existing in existing_ids
    ]

    existing_ids = [  # Drop if no ID column
        df_existing.select(pl.col('id').cast(schema.VARIANT['id'])) for df_existing in existing_ids
            if 'id' in df_existing.collect_schema().names()
    ]

    df_existing = (
        (
            pl.concat(existing_ids)
                if existing_ids else pl.LazyFrame([], schema={'id': schema.VARIANT['id']})
        )
        .filter(
            pl.col('id').is_not_null()
        )
        .with_columns(
            (
                pl.col('id').str.extract_groups(r'([^.]*)(\.([0-9]+))?$')
                .struct.rename_fields(['_id_base', '_', '_id_version'])
                .struct.field('_id_base', '_id_version')
            )
        )
        .with_columns(
            pl.col('_id_version').cast(pl.Int32).fill_null(0)
        )
        .group_by('_id_base')
        .agg(
            (pl.col('_id_version').max() + 1).alias('_id_version_existing')
        )
    )

    return (
        df
        .join(
            df_existing,
            on='_id_base',
            how='left',
        )
        .with_columns(
            pl.col('_id_version') + pl.col('_id_version_existing').fill_null(0)
        )
        .with_columns(
            pl.when(pl.col('_id_version') > 0)
            .then(pl.concat_str(pl.col('_id_base'), pl.lit('.'), pl.col('_id_version').cast(pl.String)))
            .otherwise(pl.col('_id_base'))
            .alias('id')
        )
        .select(col_list)
    )


def apply_trim_filter(
        df,
        df_trim
) -> pl.LazyFrame:
    """Apply alignment trimming filters to a variant table.

    :param df: Variant table.
    :param df_trim: Alignment trimming table.
    """

    df = (
        df
        .drop('_index', strict=False)
        .with_row_index('_index')
    )

    df_filter = (
        df
        .with_columns(
            pl.col('align_index').explode().alias('align_index')
        )
        .join(
            df_trim.select(['align_index', 'qry_pos', 'qry_end', 'filter']),
            on='align_index',
            how='inner',
            suffix='_trim'
        )
        .filter(
            pl.col('qry_pos') < pl.col('qry_end_trim'),
            pl.col('qry_end') > pl.col('qry_pos_trim')
        )
        .group_by('_index')
        .agg(
            pl.col('filter_trim').implode().alias('_filter_trim')
        )
    )

    return (
        df
        .join(df_filter, on='_index', how='left')
        .with_columns(
            pl.concat_list(
                'filter',
                pl.col('_filter_trim').fill_null([]).cast(pl.List(pl.String)),
            ).list.unique().list.sort().alias('filter')
        )
        .drop('_index', '_filter_trim')
    )


def add_cpx_derived(
        df: pl.LazyFrame,
        df_segment: pl.LazyFrame,
        collect_list: dict,
) -> None:
    """Identify noncanonical variants that can be derived from partial complex events.

    Note that the final variant ID must be set before calling this function or variant IDs linking derived variants to
    their sources will be broken ("id" column in `df`).

    Calls variants of two types:

        * DEL: Skipped bases between anchoring alignments at the outer breakpoint site.
        * DUP: Duplicated bases between anchoring alignments or small aligned segments within a CPX event.

    :param df: Table of variant calls.
    :param df_segment: Variant segment table.
    :param collect_list: Dictionary of variant tables to collect.
    """

    # Base table for derived variants
    df_base = (
        df
        .select(
            *['var_index', 'chrom', 'qry_id', 'qry_rev', 'var_score'],
            *[pl.lit(None).alias(col).cast(schema.VARIANT[col]) for col in ['qry_pos', 'qry_end']],
            pl.col('id').alias('derived'),
            pl.lit('DERIVED').alias('call_source'),
            pl.col('filter').list.concat([pl.lit(['DERIVED']).cast(pl.List(pl.String))]).alias('filter'),
        )
    )


    # Anchor DEL/DUP
    df_a = (
        df_base
        .join(
            df_segment
            .filter(pl.col('is_anchor'))
            .group_by('var_index')
            .agg(
                pl.when(pl.col('is_rev').first())
                .then(pl.struct(pl.col('end').last().alias('pos'), pl.col('pos').first().alias('end')))
                .otherwise(pl.struct(pl.col('end').first().alias('pos'), pl.col('pos').last().alias('end')))
                .struct.unnest(),
                pl.col('align_index').implode(),
                pl.col('is_rev').first()
            ),
            on='var_index',
            how='inner'
        )
        .with_columns(
            (pl.col('end') - pl.col('pos')).alias('varlen'),
            pl.min_horizontal(['pos', 'end']).alias('pos'),
            pl.max_horizontal(['pos', 'end']).alias('end'),
        )
        .with_columns(
            (
                pl.when(pl.col('varlen') > 0)
                .then(pl.lit('DEL'))
                .when(pl.col('varlen') < 0)
                .then(pl.lit('DUP'))
                .otherwise(None)
            ).alias('vartype'),
            pl.col('varlen').abs(),
            (
                pl.when(pl.col('varlen') < 0)
                .then(pl.col('is_rev') ^ pl.col('qry_rev'))
                .otherwise(pl.lit(None).cast(pl.Boolean))
            ).alias('inv_dup')
        )
        .filter(pl.col('vartype').is_not_null())
    )

    # Aligned segments between anchors
    df_b = (
        df_base.drop(['qry_pos', 'qry_end', 'chrom', 'qry_id'])
        .join(
            df_segment
            .filter(~ pl.col('is_anchor') & pl.col('is_aligned'))
            .select(['var_index', 'chrom', 'pos', 'end', 'is_rev', 'qry_id', 'qry_pos', 'qry_end', 'align_index']),
            on='var_index',
            how='inner'
        )
        .with_columns(
            (pl.col('end') - pl.col('pos')).alias('varlen'),
            pl.lit('DUP').alias('vartype'),
            (pl.col('is_rev') ^ pl.col('qry_rev')).alias('inv_dup'),
            pl.concat_list([pl.col('align_index')]).alias('align_index')
        )
    )

    # Complete variant tables
    df_derived = (
        pl.concat([df_a, df_b], how='diagonal')
        .with_columns(
            id_nonsnv().alias('id')
        )
        .drop('var_index')
    )

    collect_list['insdel'].append(
        id_and_version(
            df=(
                df_derived
                .filter(pl.col('vartype').is_in(['INS', 'DEL']))
                .drop('inv_dup')
            ),
            is_snv=False,
            existing_ids=collect_list['insdel']
        )
    )

    collect_list['dup'].append(
        id_and_version(
            df=(
                df_derived
                .filter(pl.col('vartype') == 'DUP')
            ),
            is_snv=False,
            existing_ids=collect_list['dup']
        )
    )


def apply_discord_and_inner_filter(
        df: pl.LazyFrame,
        df_discord: Optional[pl.LazyFrame],
        df_inner: Optional[pl.LazyFrame],
) -> pl.LazyFrame:
    """Apply discordant and inner filters to a variant table.

    :param df: Variant table.
    :param df_discord: Discordant table ("chrom", "pos", "end", and "id' where "id" is the variant at this locus).
    :param df_inner: Inner table ("align_index", "id" where "id" is the variant the alignment index belongs to).

    :return: Variant table with "inner" and "discord" columns with filters added for all non-empty items in either
        field.
    """

    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    if isinstance(df_discord, pl.DataFrame):
        df_discord = df_discord.lazy()

    if isinstance(df_inner, pl.DataFrame):
        df_inner = df_inner.lazy()

    # Prepare align_index
    df = (
        df
        .drop(['_index', 'discord', 'inner'], strict=False)
        .with_row_index('_index')
        # .with_columns(
        #     pl.col('align_index').list.first()
        # )
    )

    # Create inner columns
    if df_inner is not None:
        inner_col = (
            df
            .with_columns(
                pl.col('align_index').explode()
            )
            .join(
                (
                    df_inner
                    .select(
                        pl.col('align_index'),
                        pl.col('id').alias('inner')
                    )
                ),
                on='align_index',
                how='inner'
            )
            .group_by('_index')
            .agg(
                pl.col('inner').implode()
            )
        )

        df = (
            df
            .join(inner_col, on='_index', how='left')
            .with_columns(
                pl.col('inner').cast(schema.VARIANT['inner'])
            )
        )

    else:
        df = df.with_columns(pl.lit(None).cast(schema.VARIANT['inner']).alias('inner'))

    # Create discord column
    if df_discord is not None:
        discord_col = (
            df
            .filter(pl.col('inner').is_null())
            .join(
                df_discord.select(pl.all().name.prefix('_')),
                left_on='chrom',
                right_on='_chrom',
                how='left'
            )
            .filter(
                pl.col('pos') < pl.col('_end'),
                pl.col('end') > pl.col('_pos')
            )
            .group_by('_index')
            .agg(
                pl.col('_id').implode().alias('discord')
            )
        )

        df = (
            df
            .join(discord_col, on='_index', how='left')
            .with_columns(
                pl.col('discord').cast(schema.VARIANT['discord'])
            )
        )

    else:
        df = df.with_columns(pl.lit(None).cast(schema.VARIANT['discord']).alias('discord'))

    # Update filters
    df = (
        df
        .with_columns(
            pl.col('filter').list.concat(
                pl.when(pl.col('discord').is_not_null())
                .then(pl.lit(['DISCORD']))
                .otherwise(pl.lit([]))
            )
        )
        .with_columns(
            pl.col('filter').list.concat(
                pl.when(pl.col('inner').is_not_null())
                .then(pl.lit(['INNER']))
                .otherwise(pl.lit([]))
            )
        )
    )

    # Create empty discord and inner columns
    if df_discord is not None:
        df = df.with_columns(pl.col('discord').fill_null(pl.lit([])).cast(schema.VARIANT['discord']))

    if df_inner is not None:
        df = df.with_columns(pl.col('inner').fill_null(pl.lit([])).cast(schema.VARIANT['inner']))

    return df
