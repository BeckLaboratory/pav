"""Polars expressions used by variant calling routines."""

__all__ = [
    'id_snv',
    'id_nonsnv',
    # 'id',
    'id_version',
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


def id_version() -> pl.Expr:
    """De-duplicate IDs by appending an integer to ID strings.

    The first appearance of an ID is never modified. The second appearance of an ID gets ".1" appended, the third ".2",
    and so on.

    If any variant IDs are already versioned, then versions are stripped.

    :returns: An expression for versioning variant IDs.
    """

    expr_id = pl.col('id').str.replace(r'\.[0-9]*$', '')

    expr_version = (
        pl.col('filter').list.len()
        .rank(method='ordinal')
        .over(expr_id)
        - 1
    )

    return (
        pl.when(expr_version > 0)
        .then(pl.concat_str(expr_id, pl.lit('.'), expr_version.cast(pl.String)))
        .otherwise(expr_id)
        .alias('id')
    )
