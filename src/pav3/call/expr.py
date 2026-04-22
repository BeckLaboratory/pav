"""Polars expressions used by variant calling routines."""

# TODO: Switch to agglovar expressions

__all__ = [
    'id_snv',
    'id_nonsnv',
    'sort_expr',
    'qry_region_expr',
    'id_expr',
]

import polars as pl


def id_snv() -> pl.Expr:
    """Generate SNV IDs.

    :return: Expression for generating the ID column.
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

    :return: Expression for generating the ID column.
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
    """Arguments to `sort()` for variant tables.

    :return: Sort expression.
    """
    return [
        pl.col('chrom'),
        pl.col('pos'),
        pl.col('end'),
        pl.col('filter').list.len(),
    ] + (
        [pl.col('id')] if has_id else []
    )

def qry_region_expr() -> pl.Expr:
    """Get query region string from `qry_id`, `qry_pos`, and `qry_end` columns.

    :return: Query region expression.
    """
    return pl.concat_str(
        'qry_id', pl.lit(':'), pl.col('qry_pos') + 1, pl.lit('-'), pl.col('qry_end')
    ).alias('qry_region')

def id_expr() -> pl.Expr:
    """Generate variant IDs for any variant type.

    :return: ID expression.
    """
    id_snv_expr = id_snv()
    id_nonsnv_expr = id_nonsnv()

    return (
        pl.when(pl.col('vartype').str.to_uppercase() == 'SNV')
        .then(id_snv_expr)
        .otherwise(id_nonsnv_expr)
        .alias('id')
    )
