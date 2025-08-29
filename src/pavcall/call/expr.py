"""
Polars expressions used by variant calling routines.
"""

from typing import Optional

import polars as pl

def id_snv() -> pl.Expr:
    """
    Expression for the ID column for SNVs.

    Returns:
        Expression for generating the ID column.
    """

    return pl.concat_str(
        pl.col('chrom'),
        pl.lit('-'),
        pl.col('pos') + 1,
        pl.lit('-SNV-'),
        pl.col('ref').str.to_uppercase(),
        pl.col('alt').str.to_uppercase(),
    )


def id_nonsnv() -> pl.Expr:
    """
    Expression for the ID column for non-SNV variants.

    Returns:
        Expression for generating the ID column.
    """

    return pl.concat_str(
        pl.col('chrom'),
        pl.lit('-'),
        pl.col('pos') + 1,
        pl.lit('-'),
        pl.col('vartype').str.to_uppercase(),
        pl.lit('-'),
        pl.col('varlen')
    )


def id() -> pl.Expr:
    """
    Expression for the ID column of any variant type.

    Returns:
        Expression for generating the ID column.
    """

    return (
        pl.when(
            pl.col('vartype').str.to_uppercase() == 'SNV')
        .then(id_snv())
        .otherwise(id_nonsnv())
    )


def id_version() -> pl.Expr:
    """
    De-duplicate IDs by appending an integer to ID strings.

    The first appearance of an ID is never modified. The second appearance of an ID gets ".1" appended, the third ".2",
    and so on.

    If any variant IDs are already versioned, then versions are stripped.

    Returns:
        An expression for versioning variant IDs.
    """

    expr_id = pl.cum_count('id').str.replace('\..*$', '')
    expr_version = (expr_id.cum_count() - 1).over(expr_id)

    return (
        pl.when(
            expr_version > 0
        )
        .then(
            pl.concat_str(expr_id, pl.lit('.'), expr_version)
        )
    )
