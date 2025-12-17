"""Polars expressions used by variant calling routines."""

__all__ = [
    'id_snv',
    'id_nonsnv',
]

import polars as pl


def id_snv() -> pl.Expr:
    """Generate SNV IDs.

    :returns: Expression for generating the ID column.
    """
    return (
        pl.concat_str(
            pl.col('chrom'),
            pl.lit('-'),
            pl.col('pos') + 1,
            pl.lit('-SNV-'),
            pl.col('ref').str.to_uppercase(),
            pl.col('alt').str.to_uppercase(),
        )
        .alias('id')
    )


def id_nonsnv() -> pl.Expr:
    """Generate non-SNV IDs.

    :returns: Expression for generating the ID column.
    """
    return (
        pl.concat_str(
            pl.col('chrom'),
            pl.lit('-'),
            (pl.col('pos') + 1).cast(pl.String),
            pl.lit('-'),
            pl.col('vartype').str.to_uppercase(),
            pl.lit('-'),
            pl.col('varlen').cast(pl.String)
        )
        .alias('id')
    )


def sort_expr(has_id: bool = True) -> list[pl.Expr]:
    """Arguments to `sort()` for variant tables."""
    return [
        pl.col('chrom'),
        pl.col('pos'),
        pl.col('end'),
        pl.col('filter').list.len(),
    ] + (
        [pl.col('id')] if has_id else []
    )

# def id() -> pl.Expr:
#     """Generate variant IDs for any variant type.
#
#     :returns: ID expression.
#     """
#     return (
#         pl.when(pl.col('vartype').str.to_uppercase() == 'SNV')
#         .then(id_snv())
#         .otherwise(id_nonsnv())
#         .alias('id')
#     )
