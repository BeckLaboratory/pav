"""Transform variant calls from Parquet files to VCF files."""

from collections.abc import Iterable

import polars as pl
import pysam

def write_vcf(
        tables: Iterable[pl.LazyFrame | pl.DataFrame]
) -> pl.LazyFrame:
    table_list = [_init_table(df) for df in tables]

def _set_fields_snv(
        df: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Set REF, ALT, and POS for SNVs.

    :param df: Variant table (SNVs only).

    :returns: Table with VCF columns set.
    """
    return (
        df.with_columns(
            pl.col('ref').alias('REF'),
            pl.col('alt').alias('ALT'),
            (pl.col('pos') + 1).alias('POS'),
        )
    )


def _set_fields_inv(
        df: pl.LazyFrame
) -> pl.LazyFrame:

    return (
        df.with_columns(

        )
    )

def _set_ref_position_non_snv(
        df: pl.LazyFrame,
        ref_fa: str,
) -> pl.LazyFrame:

    df = (
        df.with_columns(
            (pl.col('pos') > 0).alias('_ref_base_left')
        )
        .with_columns(
            (
                pl.when('_ref_base_left')
                .then(pl.col('pos') - 1)
                .otherwise('end')
            ).alias('_ref_base_pos')
        )
    )

    with pysam.FastaFile(ref_fa) as ref_file:
        seq_ref = [
            ref_file.fetch(str(vals['chrom']), int(vals['_ref_base_pos']), int(vals['_ref_base_pos'] + 1))
                for vals in df.select('chrom', '_ref_base_pos').collect().rows(named=True)
        ]



            .with_columns(
                pl.struct([pl.col('chrom'), pl.col('_ref_base_pos')])
                .map_elements(
                    lambda vals: ref_file.fetch(str(vals['chrom']), int(vals['_ref_base_pos']), int(vals['_ref_base_pos'] + 1)),
                    return_dtype=pl.String
                ).cast(pl.String).alias('_ref_base')
            )
        )

    return df

def _init_vcf_fields(
        df: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    """Initialize VCF fields common across all variant types (does not set REF, ALT, or POS).

    :param df: Variant table.
    """

    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    return (
        df.with_columns(
            pl.col('chrom').alias('#CHROM'),
            pl.col('id').alias('ID'),
            pl.lit('.').alias('QUAL'),
            (
                pl.when(pl.col('filter').is_not_null())
                .then(pl.col('filter').list.join(';'))
                .otherwise(pl.lit('PASS'))
            ).alias('FILTER'),
            pl.lit([]).cast(pl.List(pl.String)).alias('_info'),
            pl.lit([]).cast(pl.List(pl.Struct({'fmt': pl.String, 'sample': pl.String}))).alias('_sample')
        )
        .with_columns(
            pl.col('_info').list.concat([
                pl.concat_str(pl.lit('ID='), 'id'),
                pl.concat_str(pl.lit('QRY_REGION='), pl.col('chrom'), pl.lit(':'), pl.col('qry_pos') + 1, pl.lit('-'), pl.col('qry_end')),
                # pl.concat_str(pl.lit('QRY_STRAND='), pl.when(pl.col('is_rev')).then('-').otherwise('+')),
                pl.concat_str(pl.lit('ALIGN_INDEX='), pl.col('align_index').cast(pl.List(pl.String)).list.join(','))
            ])
        )
    )

# def _init_table():
#     df = (
#         df.with_columns(
#             pl.col('chrom').alias('#CHROM'),
#             # POS
#             pl.col('id').alias('ID'),
#             # REF
#             # ALT
#             pl.lit('.').alias('QUAL'),
#             (
#                 pl.when(pl.col('filter').is_not_null())
#                 .then(pl.col('filter').list.join(';'))
#                 .otherwise(pl.lit('PASS'))
#             ).alias('FILTER'),
#             # INFO
#             # FORMAT
#             # Sample
#         )
#     )

# def _init_table_other(
#
# ):
#     df = (
#         df.with_columns(
#             (pl.col('vartype').is_in(['INV', 'CPX', 'DUP'])).alias('_is_sym'),
#             (pl.col('vartype') == 'SNV').alias('_is_snv'),
#             (pl.col('pos') > 0).alias('_ref_left')  # Append reference base to left if True (most variants), move to end if not (variant at position 0)
#         )
#     )


def _ref_base(df, ref_fa):
    """
    Get reference base preceding a variant (SV, indel) or at the point change (SNV).

    :param df: Variant dataframe as BED.
    :param ref_fa: Reference file.
    """

    # Get the reference base location
    with pysam.FastaFile(ref_fa) as ref_file:

        df = (
            df.with_columns(
                (
                    pl.when(pl.col('_ref_left'))
                    .then(pl.col('pos') - 1)
                    .otherwise(pl.col('end'))
                ).alias('_ref_base_loc')
            )
            .with_columns(
                pl.when(pl.col('_is_snv'))
                .then(pl.col('_ref_base_loc') + 1)
                .otherwise(pl.col('_ref_base_loc'))
            ).alias('_ref_base_loc')
            .with_columns(
                pl.struct(['chrom', '_ref_base_loc'])
                .apply(
                    lambda vals: ref_file.fetch(vals['chrom'], vals['_ref_base_loc'], vals['_ref_base_loc'] + 1)
                ).alias('_ref_base')
            )
            .drop('_ref_base_Loc')
        )

    # Open and update records
    with pysam.FastaFile(ref_fa) as ref_file:
        for index, row in df.iterrows():

            if row['SVTYPE'] in {'INS', 'DEL', 'INSDEL', 'DUP', 'INV'}:
                yield ref_file.fetch(row['#CHROM'], row['POS'] + (-1 if row['POS'] > 0 else 0), row['POS']).upper()

            elif row['SVTYPE'] == 'SNV':
                if 'REF' in row:
                    yield row['REF']
                else:
                    yield ref_file.fetch(row['#CHROM'], row['POS'], row['POS'] + 1).upper()

            else:
                raise RuntimeError('Unknown variant type: "{}" at index {}'.format(row['VARTYPE'], index))
