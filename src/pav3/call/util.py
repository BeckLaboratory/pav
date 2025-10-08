"""Variant calling utilities."""

__all__ = [
    'filter_trim'
]

import polars as pl
import polars.selectors as cs


def filter_trim(
        df: pl.DataFrame | pl.LazyFrame,
        df_align: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    """Filter a variant table by trimmed alignments.

    Retain only variants that are not fully within non-trimmed regions.

    :param df: Variant table.
    :param df_align: Alignment table this variant table was generated from. The align_index field in the variant table
        must match the alignment record the variant was generated from.

    :returns: Filtered variant table.
    """
    if not isinstance(df, pl.LazyFrame):
        df = df.lazy()

    if not isinstance(df_align, pl.LazyFrame):
        df_align = df_align.lazy()

    return (
        df.join(
            df_align.select(
                pl.col('align_index'),
                pl.col('qry_pos').alias('_qry_pos_r'),
                pl.col('qry_end').alias('_qry_end_r'),
            ),
            on='align_index',
            how='left'
        )
        .filter(
            pl.col('_qry_pos_r').is_not_null() & (
                (pl.col('qry_end') < pl.col('_qry_pos_r')) | (pl.col('qry_pos') > pl.col('_qry_end_r'))
            )
        )
        .drop(
            cs.starts_with('_')
        )
    )
