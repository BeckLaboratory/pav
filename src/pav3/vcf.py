"""Transform variant calls from Parquet files to VCF files."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import datetime
from pathlib import Path
import threading
from typing import Optional

import agglovar
import polars as pl
import polars.selectors as cs
import pysam

from . import __version__
from .const import FILTER_REASON

# Note on number fields (PAV does not use most of these)
#
# Number fields:
# * Int: Exact number
# * A: One per ALT allele
# * R: One per allele including REF
# * G: One per each possible genotype
# * .: Varies, unknown, or unbounded
#
# Format fields have additional Number options:
# * LA: A except only ALT alleles in LAA are considered present.
# * LR: R except only ALT alleles in LAA are considered present.
# * LG: G except only ALT alleles in LAA are considered present.
# * P: One value for each allele value in GT.
# * M: One value for each possible base modification.

@agglovar.meta.decorators.immutable
class InfoField:
    name: str = agglovar.meta.descriptors.CheckedString(match=r'[A-Z]([A-Z_]*[A-Z])?')
    number: str = agglovar.meta.descriptors.CheckedString(match=r'([1-9][0-9]*)|[ARG.]')
    type_: str = agglovar.meta.descriptors.CheckedString(match={'Integer', 'Float', 'Flag', 'Character', 'String'})
    description: str = agglovar.meta.descriptors.CheckedString(min_len=1, strip=True)

    def __init__(self, name, number, type_, description):
        self.name = str(name)
        self.number = str(number)
        self.type_ = str(type_)
        self.description = str(description)

        if self.type_ == 'Flag' and self.number != '0':
            raise ValueError(f'Flag type must have number 0: Number={self.number}')

    def __str__(self) -> str:
        return f'##INFO=<ID={self.name},Number={self.number},Type={self.type_},Description="{self.description}">'

@agglovar.meta.decorators.immutable
class FilterField:
    name: str = agglovar.meta.descriptors.CheckedString(match=r'[A-Z]([A-Z_]*[A-Z])?')
    description: str = agglovar.meta.descriptors.CheckedString(min_len=1, strip=True)

    def __init__(self, name, description):
        self.name = str(name)
        self.description = str(description)

    def __str__(self) -> str:
        return f'##FILTER=<ID={self.name},Description="{self.description}">'

@agglovar.meta.decorators.immutable
class FormatField:
    name: str = agglovar.meta.descriptors.CheckedString(match=r'[A-Z]([A-Z_]*[A-Z])?')
    number: str = agglovar.meta.descriptors.CheckedString(match=r'([1-9][0-9]*)|[ARGPM.]|L[ARG]')
    type_: str = agglovar.meta.descriptors.CheckedString(match={'Integer', 'Float', 'Character', 'String'})
    description: str = agglovar.meta.descriptors.CheckedString(min_len=1, strip=True)

    def __init__(self, name, number, type_, description):
        self.name = str(name)
        self.number = str(number)
        self.type_ = str(type_)
        self.description = str(description)

    def __str__(self) -> str:
        return f'##FORMAT=<ID={self.name},Number={self.number},Type={self.type_},Description="{self.description}">'

@agglovar.meta.decorators.immutable
class AltField:
    name: str = agglovar.meta.descriptors.CheckedString(match=r'[A-Z]([A-Z:_]*[A-Z])?')
    description: str = agglovar.meta.descriptors.CheckedString(min_len=1, strip=True)

    def __init__(self, name, description):
        self.name = str(name)
        self.description = str(description)

    def __str__(self) -> str:
        return f'##ALT=<ID={self.name},Description="{self.description}">'

VCF_VERSION = '4.5'

INFO_FIELDS = [
    InfoField(
        'ID', '1', 'String',
        'Unique variant ID',
    ),
    InfoField(
        'SVLEN', '.', 'Integer',
        'Variant length',
    ),
    InfoField(
        'END', '.', 'Integer',
        'Variant end position',
    ),
    InfoField(
        'SUBTYPE', '.', 'String',
        'Variant subtype (TD for tandem duplications)',
    ),
    InfoField(
        'CPX_REF_TRACE', '.', 'String',
        'Reference trace for complex variants',
    ),
    InfoField(
        'CPX_QRY_TRACE', '.', 'String',
        'Query trace for complex variants',
    ),
    InfoField(
        'QUERY', '.', 'String',
        'Query sequence position',
    ),
    InfoField(
        'QUERYREV', '.', 'Character',
        'Query sequence orientation vs the reference (+ or -, * if unknown',
    ),
    InfoField(
        'SEQ', '.', 'String',
        'Variant sequence.',
    ),
    InfoField(
        'DUP', '.', 'String',
        'Comma-separated list of regions duplicated by this variant, each region in format '
        '\\"chrom:pos-end\\" (1-base, closed coordinates)'
    ),
    InfoField(
        'HOMREF', '.', 'String',
        'Reference homology as a comma-separated list of two integers (upstream homology, downstream homology). '
        'Prefer reference homology for deletions.'
    ),
    InfoField(
        'HOMQRY', '.', 'String',
        'Query homology as a comma-separated list of two integers (upstream homology, downstream homology). '
        'Prefer query homology for insertions.'
    ),
]

FILTER_FIELDS = [
    FilterField(
        'PASS', 'All filters passed',
    ),
] + [
    FilterField(name, description)
    for name, description in FILTER_REASON.items()
]

FORMAT_FIELDS = [
    FormatField(
        'GT', '1', 'String',
        'Genotype',
    ),
]

ALT_FIELDS = [
    AltField('INS', 'Insertion'),
    AltField('DEL', 'Deletion'),
    AltField('INV', 'Inversion'),
    AltField('DUP', 'Duplication'),
    AltField('CPX', 'Complex variant'),
]


def get_headers(
        ref_filename: Optional[str | Path] = None,
        df_ref_info: Optional[pl.DataFrame] = None,
) -> list[str]:

    # Lead headers
    headers = [
        f'##fileformat=VCFv{VCF_VERSION}',
        f'##filedate={datetime.datetime.now().strftime("%Y%m%d")}',
        f'##source=PAVv{__version__}',
    ]

    # Reference
    if ref_filename is not None:
        # Only filename, do not leak filesystem paths for security reasons
        headers.append(
            f'##reference=file://{Path(ref_filename).name}'
        )

    # Reference contigs
    if df_ref_info is not None:
        headers.extend(
            f'##contig=<ID={chrom},length={length},md5={md5}>'
            for chrom, length, md5 in df_ref_info.select(['chrom', 'len', 'md5']).iter_rows()
        )

    # ALT headers
    headers.extend(
        str(alt_field) for alt_field in ALT_FIELDS
    )

    # INFO headers
    headers.extend(
        str(info_field) for info_field in INFO_FIELDS
    )

    # FILTER headers
    headers.extend(
        str(filter_field) for filter_field in FILTER_FIELDS
    )

    # FORMAT headers
    headers.extend(
        str(format_field) for format_field in FORMAT_FIELDS
    )

    return headers

def init_vcf_fields(
        df: pl.LazyFrame,
        ref_fa: Optional[str] = None,
        use_sym: Optional[bool] = None,
        vartype: Optional[str] = None,
) -> pl.LazyFrame:
    """Initialize VCF fields.

    If `hap_source` and `hap_callable` are provided, genotypes will be added to the table. If one is provided and
    the other is None, an error is raised.

    :param df: Variant table with an "_index" column.
    :param ref_fa: Reference FASTA filename. Required for non-SNV variants.
    :param use_sym: Whether to use symbolic alternate records. If None, INS, DEL, and SNV ore not
        symbolic, and all other variant types are.
    """
    vartype_set = set(
        df.select(pl.col('vartype').unique()).collect().to_series()
    )

    if 'SNV' in vartype_set and len(vartype_set) > 1:
        raise ValueError(
            f'Cannot mix SNV variants with other variant types: '
            f'Found variant types {", ".join(vartype_set)}'
        )

    is_snv = 'SNV' in vartype_set

    if is_snv and use_sym:
        raise ValueError('Variant type "SNV" cannot use symbolic ALTs')

    if use_sym is None:

        no_sym_set = vartype_set & {'INS', 'DEL', 'SNV'}
        is_sym_set = vartype_set - {'INS', 'DEL', 'SNV'}

        if not (no_sym_set or is_sym_set):
            use_sym = False

        elif no_sym_set and is_sym_set:
            raise ValueError(
                f'Unable to automatically determine whether to use symbolic ALTs or sequence ALTs given variant types '
                f'using sequence ALTs (INS, DEL, SNV) and symbolic ALTs (all others): '
                f'Found variant types {", ".join(vartype_set)}'
            )
        else:
            use_sym = bool(is_sym_set)

    if use_sym and is_snv:
        raise ValueError('Variant type "SNV" cannot use symbolic ALTs')

    if not use_sym and (vartype_set - {'INS', 'DEL', 'SNV'}):
        raise ValueError(
            f'Can only use sequence ALTs for INS, DEL, and SNV: '
            f'Found variant types {", ".join(vartype_set)}'
        )

    # Get POS, REF, and ALT
    if is_snv:
        df_pos_ref_alt = vcf_fields_snv(df)

    elif use_sym:
        if ref_fa is None:
            raise ValueError('Missing reference FASTA filename: Required for symbolic ALTs')

        df_pos_ref_alt = vcf_fields_sym(df, ref_fa)

    else:
        if ref_fa is None:
            raise ValueError('Missing reference FASTA filename: Required for non-SNV sequence ALTs')

        df_pos_ref_alt = vcf_fields_seq_insdel(df, ref_fa)

    # Initialize fields
    df_vcf = (
        df
        .with_columns(
            '_index',
            pl.col('chrom').alias('_vcf_chrom'),
            pl.col('id').alias('_vcf_id'),
            pl.lit('.').alias('_vcf_qual'),
            (
                pl.when(pl.col('filter').list.len() > 0)
                .then(pl.col('filter').list.join(separator=';'))
                .otherwise(pl.lit('PASS'))
                .cast(pl.String)
                .alias('_vcf_filter')
            ),
            (
                pl.lit([])
                .cast(pl.List(pl.String))
                .alias('_vcf_info')
            ),
            (
                pl.lit([])
                .cast(pl.List(pl.String))
                .alias('_vcf_format')
            )
        )
    )

    # Add POS, REF, and ALT
    return (
        df_vcf
        .drop(['_vcf_pos', '_vcf_ref', '_vcf_alt', '_is_sym_alt'], strict=False)
        .join(
            (
                df_pos_ref_alt
                .select('_index', '_vcf_pos', '_vcf_ref', '_vcf_alt', '_is_sym_alt')
            ), on='_index', how='left'
        )
    )


def vcf_fields_seq_insdel(
        df: pl.LazyFrame,
        ref_fa: str,
) -> pl.LazyFrame:
    """Set POS, REF, and ALT VCF fields for insertions and deletions (non-symbolic).

    VCF records where "REF" and "ALT" contain sequences (i.e. not symbolic ALTs like "<INS>")
    require a reference base immediately upstream or downstream of the variant to set the context.
    This function sets these fields and extracts the extra reference base.

    The reference base in field "_ref_base" is the base immediately before the variant (if variant
    position > 0) or immediately after (variant position == 0). Field "_ref_base_left" is a boolean
    flag indicating whether "_ref_base" is the base before the variant or after. In both cases,
    "_ref_base_pos" is the position of the variant in the reference base in reference
    coordinates (0-based).

    Returned table has fields:

    * _index: Index of variant.
    * _vcf_pos: Variant position for VCF field "POS".
    * _vcf_ref: Reference sequence for VCF field "REF".
    * _vcf_alt: Alternate sequence for VCF field "ALT".
    * _ref_base: Reference base.
    * _ref_base_pos: Position of variant in reference base in reference coordinates (0-based).
    * _ref_base_left: Whether the reference base is the base before the variant (True) or
      after (False). This is used to determine whether to put the reference base before or
      after the sequence when constructing the REF and ALT fields.
    * _is_sym_alt: Whether the alternate sequence is a symbolic alternate (always False).

    :param df: Variant table with an "_index" column.
    :param ref_fa: Reference FASTA filename.

    :returns: Variant table with reference base and position fields added.
    """

    if ref_fa is None:
        raise ValueError('Missing reference FASTA filename')

    ref_fa = str(ref_fa)

    # Get position of the reference base
    df_fields = (
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

    # Get fields
    with _LockingPysam(ref_fa) as ref_file:
        df_fields = (
            df_fields
            .select(
                '_index',
                (
                    pl.struct([pl.col('chrom'), pl.col('_ref_base_pos')])
                    .map_elements(
                        lambda vals: ref_file.fetch(str(vals['chrom']), int(vals['_ref_base_pos']), int(vals['_ref_base_pos'] + 1)),
                        return_dtype=pl.String
                    )
                    .cast(pl.String)
                    .alias('_ref_base')
                ),
                '_ref_base_pos',
                '_ref_base_left',
                'pos',
                'vartype',
                'seq'
            )
            .collect().lazy()  # Must collect while ref_file is open
        )

    # Set REF and ALT fields
    return (
        df_fields
        .with_columns(  # Set REF and ALT fields
            (
                pl.when(pl.col('vartype') == 'INS')
                .then(pl.lit(''))
                .when(pl.col('vartype') == 'DEL')
                .then(pl.col('seq'))
                .cast(pl.String)
                .alias('_vcf_ref')
            ),
            (
                pl.when(pl.col('vartype') == 'INS')
                .then(pl.col('seq'))
                .when(pl.col('vartype') == 'DEL')
                .then(pl.lit(''))
                .cast(pl.String)
                .alias('_vcf_alt')
            )
        )
        .select(
            '_index',
            (
                pl.when(pl.col('_ref_base_left'))
                .then(pl.col('pos'))
                .otherwise(pl.col('pos') + 1)
                .alias('_vcf_pos')
            ),
            (
                pl.when(pl.col('_ref_base_left'))
                .then(pl.concat_str('_ref_base', '_vcf_ref'))
                .otherwise(pl.concat_str('_vcf_ref', '_ref_base'))
                .cast(pl.String)
                .alias('_vcf_ref')
            ),
            (
                pl.when(pl.col('_ref_base_left'))
                .then(pl.concat_str('_ref_base', '_vcf_alt'))
                .otherwise(pl.concat_str('_vcf_alt', '_ref_base'))
                .cast(pl.String)
                .alias('_vcf_alt')
            ),
            pl.lit(False).alias('_is_sym_alt'),
            '_ref_base',
            '_ref_base_pos',
            '_ref_base_left',
        )
    )


def vcf_fields_sym(
    df: pl.LazyFrame,
    ref_fa: str,
) -> pl.LazyFrame:
    """Get POS, REF, and ALT fields for symbolic alternate records.

    Symbolic alts such as "<INS>" and "<DEL>" can be set for the ALT VCF field instead of exact
    sequences. In these cases, the SEQ info field should also be included (not handled by this
    function).

    * _index: Index of variant.
    * _vcf_pos: Variant position for VCF field "POS".
    * _vcf_ref: Reference sequence for VCF field "REF".
    * _vcf_alt: Symbolic alternate representation.
    * _is_sym_alt: Whether the alternate sequence is a symbolic alternate (always True).

    :param df: Variant table with an "_index" column.
    :param ref_fa: Reference FASTA filename.

    :returns: LazyFrame with VCF fields set.
    """
    with _LockingPysam(ref_fa) as ref_file:
        return (
            df
            .select(
                '_index',
                pl.col('pos').alias('_vcf_pos'),
                (
                    pl.struct([pl.col('chrom'), pl.col('pos')])
                    .map_elements(
                        lambda vals: ref_file.fetch(str(vals['chrom']), int(vals['pos']), int(vals['pos'] + 1)),
                        return_dtype=pl.String
                    )
                    .cast(pl.String)
                    .alias('_vcf_ref')
                ),
                (
                    pl.concat_str(
                        pl.lit('<'),
                        pl.col('vartype').str.to_uppercase(),
                        pl.lit('>')
                    )
                    .alias('_vcf_alt')
                ),
                pl.lit(True).alias('_is_sym_alt'),
            )
            .collect().lazy()  # Must collect while ref_file is open
        )


def vcf_fields_snv(
        df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Set REF and ALT for VCF fields for SNVs.

    VCF records where "REF" and "ALT" contain sequences (i.e. not symbolic ALTs like "<INS>")
    require a reference base immediately upstream or downstream of the variant to set the context.
    This function sets these fields and extracts the extra reference base.

    The reference base in field "_ref_base" is the base immediately before the variant (if variant
    position > 0) or immediately after (variant position == 0). Field "_ref_base_left" is a boolean
    flag indicating whether "_ref_base" is the base before the variant or after. In both cases,
    "_ref_base_pos" is the position of the variant in the reference base in reference
    coordinates (0-based).

    Returned table has fields:

    * _index: Index of variant.
    * _vcf_pos: Variant position for VCF field "POS".
    * _vcf_ref: Reference sequence for VCF field "REF".
    * _vcf_alt: Alternate sequence for VCF field "ALT".
    * _is_sym_alt: Whether the alternate sequence is a symbolic alternate (always False).

    :param df: Variant table with an "_index" column.

    :returns: Variant table with reference base and position fields added.
    """

    return (
        df
        .select(
            '_index',
            (pl.col('pos') + 1).alias('_vcf_pos'),
            pl.col('ref').alias('_vcf_ref'),
            pl.col('alt').alias('_vcf_alt'),
            pl.lit(False).alias('_is_sym_alt'),
        )
    )


def gt_column(
        df: pl.LazyFrame,
        hap_source: dict[str, int],
        hap_callable: dict[str, pl.LazyFrame],
        col_name: str = 'gt',
        separator: str = '|',
):
    """Get a table with formatted genotypes.

    :param df: Variant table with an "_index" column.
    :param hap_source: A dictionary mapping haplotype names (keuys) to the index of the haplotype in merged order
        (0 is the first merged haplotype, 1 is the second, etc).
    :param hap_callable: A dictionary mapping haplotype names to a polars LazyFrame of callable regions for that
        haplotype.
    :param col_name: Name of the genotype column in the output table.
    :param separator: Genotype separator character ("|" for phased, "/" for unphased).

    :return: A polars LazyFrame with two columns, "_index" and the genoytpe string (column name is set by `col_name`).
    """

    if len(hap_source) == 0:
        raise ValueError(f'No haplotypes to get genotypes for')

    if '_index' not in df.collect_schema():
        df = df.with_row_index('_index')

    # Start with an index, add genotype columns
    df_gt = df.select('_index')

    for hap in hap_source.keys():
        hap_index = hap_source[hap]

        # Determine if each value matches this haplotype.
        callable_hap = (
            agglovar.bed.intersect.as_proportion(
                df,
                hap_callable[hap],
                'callable_hap'
            )
            .with_columns(
                pl.col('callable_hap') >= 0.5
            )
        )

        df_hap = (
            df.select(
                '_index', 'chrom', 'pos', 'end',
                (
                    pl.col('mg_src')
                    .list.eval(
                        pl.element().struct.field('index')
                    )
                    .list.contains(hap_index)
                    .alias('in_hap')
                ),
            )
            .join(
                callable_hap, on='_index', how='left'
            )
            .with_columns(
                pl.col('callable_hap').fill_null(False)
            )
            .select(
                '_index',
                pl.when(pl.col('in_hap'))
                .then(pl.lit('1'))
                .when(pl.col('callable_hap'))
                .then(pl.lit('0'))
                .otherwise(pl.lit('.'))
                .cast(pl.String)
                .alias(hap)
            )
        )

        df_gt = df_gt.join(df_hap, on='_index', how='left')

    # Join tables and get haplotypes
    return (
        df_gt
        .select(
            '_index',
            pl.concat_str(cs.exclude('_index').fill_null(pl.lit('.')), separator=separator).alias(col_name)
        )
    )


def standard_info_fields(
        df: pl.LazyFrame,
) -> pl.LazyFrame:
    """Set standard INFO columns.

    Appends to the "_vcf_info" list for known INFO annotations found in variant tables.

    :param df: Variant table.

    :returns: Variant table with INFO columns added.
    """

    cols = set(df.collect_schema())

    if '_vcf_info' not in cols:
        df = df.with_columns(
            pl.lit([])
            .cast(pl.List(pl.String))
            .alias('_vcf_info')
        )

    if 'id' in cols:
        df = df.with_columns(
            pl.col('_vcf_info').list.concat(
                pl.concat_str(pl.lit('ID='), 'id')
            )
        )

    if 'varlen' in cols:
        df = df.with_columns(
            pl.col('_vcf_info').list.concat(
                pl.concat_str(
                    pl.lit('SVLEN='),
                    pl.when(pl.col('vartype') != 'DEL')
                    .then('varlen')
                    .otherwise(- pl.col('varlen'))
                )
            )
        )

        df = df.with_columns(
            pl.when(pl.col('vartype').is_in({'DEL', 'INV'}))
            .then(
                pl.col('_vcf_info').list.concat(
                    pl.concat_str(
                        pl.lit('END='),
                        pl.col('_vcf_pos') + pl.col('varlen')
                    )
                )
            )
            .otherwise('_vcf_info')
            .alias('_vcf_info')
        )

    if 'varsubtype' in cols:
        df = df.with_columns(
            pl.when(pl.col('varsubtype').is_not_null())
            .then(
                pl.col('_vcf_info').list.concat(
                    pl.concat_str(pl.lit('SUBTYPE='), 'varsubtype')
                )
            )
            .otherwise('_vcf_info')
            .alias('_vcf_info')
        )

    if 'trace_ref' in cols:
        df = df.with_columns(
            pl.when(pl.col('trace_ref').is_not_null())
            .then(
                pl.col('_vcf_info').list.concat(
                    pl.concat_str(pl.lit('CPX_REF_TRACE='), 'trace_ref')
                )
            )
            .otherwise('_vcf_info')
            .alias('_vcf_info')
        )

    if 'trace_qry' in cols:
        df = df.with_columns(
            pl.when(pl.col('trace_qry').is_not_null())
            .then(
                pl.col('_vcf_info').list.concat(
                    pl.concat_str(pl.lit('CPX_QRY_TRACE='), 'trace_qry')
                )
            )
            .otherwise('_vcf_info')
            .alias('_vcf_info')
        )

    if len(cols & {'qry_id', 'qry_pos', 'qry_end'}) == 3:
        df = df.with_columns(
            pl.col('_vcf_info').list.concat(
                pl.concat_str(
                    pl.lit('QUERY='),
                    pl.col('qry_id'),
                    pl.lit(':'),
                    pl.col('qry_pos') + 1,
                    pl.lit('-'),
                    pl.col('qry_end')
                )
            )
        )

    if 'qry_rev' in cols:
        df = df.with_columns(
            pl.col('_vcf_info').list.concat(
                pl.concat_str(
                    pl.lit('QUERYREV='),
                    pl.col('qry_rev').replace_strict({True: '-', False: '+'}, default='*')
                )
            )
        )

    if '_is_sym_alt' in cols:
        if 'seq' in cols:
            df = df.with_columns(
                pl.when(pl.col('seq').is_not_null())
                .then(
                    pl.col('_vcf_info').list.concat(
                        pl.concat_str(pl.lit('SEQ='), 'seq')
                    )
                )
                .otherwise('_vcf_info')
                .alias('_vcf_info')
            )

    if 'dup' in cols:
        df = df.with_columns(
            pl.when(pl.col('dup').is_not_null() & pl.col('dup').list.len() > 0)
            .then(
                pl.col('_vcf_info').list.concat(
                    pl.concat_str(
                        pl.lit('DUP='),
                        pl.col('dup').list.eval(
                            pl.concat_str(
                                pl.element().struct.field('chrom'),
                                pl.lit(':'),
                                (pl.element().struct.field('pos') + 1).cast(pl.String),
                                pl.lit('-'),
                                pl.element().struct.field('end').cast(pl.String),
                            )
                        ).list.join(',')
                    )
                )
            )
            .otherwise('_vcf_info')
            .alias('_vcf_info')
        )

    if 'hom_ref' in cols:
        df = df.with_columns(
            pl.when(pl.col('hom_ref').is_not_null())
            .then(
                pl.col('_vcf_info').list.concat(
                    pl.concat_str(
                        pl.lit('HOMREF='),
                        pl.col('hom_ref').struct.field('up'),
                        pl.lit(','),
                        pl.col('hom_ref').struct.field('dn'),
                    )
                )
            )
            .otherwise('_vcf_info')
            .alias('_vcf_info')
        )

    if 'hom_qry' in cols:
        df = df.with_columns(
            pl.when(pl.col('hom_qry').is_not_null())
            .then(
                pl.col('_vcf_info').list.concat(
                    pl.concat_str(
                        pl.lit('HOMREF='),
                        pl.col('hom_ref').struct.field('up'),
                        pl.lit(','),
                        pl.col('hom_ref').struct.field('dn'),
                    )
                )
            )
            .otherwise('_vcf_info')
            .alias('_vcf_info')
        )

    # TODO: Adjust inner and derived annotations for merged variants.
    #
    # Merging can get variant IDs out of sync with annotated derived and inner IDs. For example, if Variant A is
    # inside variant B and the representations for variants A and B come from different haplotypes, then this annotation
    # may not make sense. For example, if Variant A is derived from haplotype "h2" and variant B derived from haplotype
    # "h1" (e.g. variant B is homozygous and A is not), then the variant IDs will be out of sync. Unless merging
    # corrects this, do not report in the VCF.
    #
    # if 'derived' in cols:
    #     df = df.with_columns(
    #         pl.when(pl.col('derived').is_not_null())
    #         .then(
    #             pl.col('_vcf_info').list.concat(
    #                 pl.concat_str(pl.lit('DERIVED='), 'derived')
    #             )
    #         )
    #         .otherwise('_vcf_info')
    #         .alias('_vcf_info')
    #     )
    #
    # if 'inner' in cols:
    #     df = df.with_columns(
    #         pl.when(pl.col('inner').list.len() > 0)
    #         .then(
    #             pl.col('_vcf_info').list.concat(
    #                 pl.concat_str(
    #                     pl.lit('INNER='),
    #                     pl.col('inner').list.join(','),
    #                 )
    #             )
    #         )
    #         .otherwise('_vcf_info')
    #         .alias('_vcf_info')
    #     )
    #
    # if 'discord' in cols:
    #     df = df.with_columns(
    #         pl.when(pl.col('discord').list.len() > 0)
    #         .then(
    #             pl.col('_vcf_info').list.concat(
    #                 pl.concat_str(
    #                     pl.lit('DISCORD='),
    #                     pl.col('inner').list.join(','),
    #                 )
    #             )
    #         )
    #         .otherwise('_vcf_info')
    #         .alias('_vcf_info')
    #     )

    return df


def reformat_vcf_table(
        df: pl.LazyFrame,
        sample_columns: Optional[Mapping[int, str]] = None,
) -> pl.LazyFrame:
    """Reformat VCF table to VCF format."""

    sample_col_exprs = [
        (
            pl.col(f'_vcf_sample_{i}')
            .list.eval(  # Replacing semicolons should not be needed, but don't break VCFs if they are present
                pl.element().str.replace_all(';', ':')
            )
        ).list.join(';').alias(sample)
        for i, sample in sample_columns.items()
    ] if sample_columns is not None else []

    return (
        df
        .select(
            pl.col('_vcf_chrom').alias('#CHROM'),
            pl.col('_vcf_pos').alias('POS'),
            pl.col('_vcf_id').fill_null('.').alias('ID'),
            pl.col('_vcf_ref').alias('REF').str.to_uppercase(),
            pl.col('_vcf_alt').alias('ALT').str.to_uppercase(),
            pl.lit('.').alias('QUAL'),
            pl.col('_vcf_filter').alias('FILTER'),
            (
                pl.col('_vcf_info')
                .list.eval(  # Replacing semicolons should not be needed, but don't break VCFs if they are present
                    pl.element().str.replace_all(';', ':')
                )
            ).list.join(';').alias('INFO'),
            pl.col('_vcf_format').list.join(';').alias('FORMAT'),
            *sample_col_exprs,
        )
    )


def get_hap_source(
        hap_list: Iterable[str],
        vcf_haplotypes: Optional[str],
) -> dict[str, int]:
    """Get a dict of haplotypes with keys in the order they should appear in the VCF genotype field.

    :param hap_list: List of haplotypes defined for this assembly.
    :param vcf_haplotypes: Optional, a string of haplotypes separated by commas to include in the VCF haplotypes. If
        defined, all haplotypes must appear in `hap_list` and the retured dictionary is subset and ordered by this list.

    :param: A dictionary of haplotype name (keys) and haplotype indices (values) that will match
    """
    hap_source = {hap: i for i, hap in enumerate(hap_list)}

    if vcf_haplotypes is not None:
        vcf_haps = [hap.strip() for hap in vcf_haplotypes.split(',') if hap.strip()]

        if (missing_hap := set(vcf_haps) - set(hap_list)):
            missing_hap = [
                f'"{hap}"' for hap in sorted(missing_hap)
            ]

            missing_n = len(missing_hap)

            missing_hap = ', '.join(missing_hap[:3]) + ('...' if missing_n > 3 else '')

            raise ValueError(
                f'Configuration item "vcf_haplotypes" defines {missing_n} haplotypes that are not in the assembly '
                f'table for "{wildcards.asm_name}": {missing_hap}'
            )

        # Filter and order hap_source to VCF order
        hap_source = {hap: hap_source[hap] for hap in vcf_haps}

    return hap_source


class _LockingPysam():
    """Serves pysam.FastaFile.fetch() under a lock

    Pysam is not thread-safe, and will fail when called through Polars. This class explicitly locks it.
    """
    filename: str

    def __init__(
            self,
            filename,
    ):
        """Create a new locking object.

        :param filename: FASTA file name.
        """
        self.filename = filename
        self.ref_file = None
        self.lock = threading.Lock()

    def __enter__(self):
        if self.ref_file is not None:
            raise RuntimeError('Ref file already open')

        self.ref_file = pysam.FastaFile(self.filename)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ref_file is None:
            raise RuntimeError('Ref file not open')

        self.ref_file.close()
        self.ref_file = None

    def fetch(
            self,
            chrom: str,
            start: int,
            end: int,
    ):
        if self.ref_file is None:
            raise RuntimeError('Ref file not open')

        with self.lock:
            return self.ref_file.fetch(chrom, start, end)


#
# Refactor
#


# def _init_vcf_fields(
#         df: pl.LazyFrame | pl.DataFrame,
# ) -> pl.LazyFrame:
#     """Initialize VCF fields common across all variant types (does not set REF, ALT, or POS).
#
#     :param df: Variant table.
#     """
#
#     if isinstance(df, pl.DataFrame):
#         df = df.lazy()
#
#     return (
#         df.with_columns(
#             pl.col('chrom').alias('#CHROM'),
#             pl.col('id').alias('ID'),
#             pl.lit('.').alias('QUAL'),
#             (
#                 pl.when(pl.col('filter').is_not_null())
#                 .then(pl.col('filter').list.join(';'))
#                 .otherwise(pl.lit('PASS'))
#             ).alias('FILTER'),
#             pl.lit([]).cast(pl.List(pl.String)).alias('_info'),
#             pl.lit([]).cast(pl.List(pl.Struct({'fmt': pl.String, 'sample': pl.String}))).alias('_sample')
#         )
#         .with_columns(
#             pl.col('_info').list.concat([
#                 pl.concat_str(pl.lit('ID='), 'id'),
#                 pl.concat_str(pl.lit('QRY_REGION='), pl.col('chrom'), pl.lit(':'), pl.col('qry_pos') + 1, pl.lit('-'), pl.col('qry_end')),
#                 # pl.concat_str(pl.lit('QRY_STRAND='), pl.when(pl.col('is_rev')).then('-').otherwise('+')),
#                 pl.concat_str(pl.lit('ALIGN_INDEX='), pl.col('align_index').cast(pl.List(pl.String)).list.join(','))
#             ])
#         )
#     )

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

#
# def _ref_base(df, ref_fa):
#     """
#     Get reference base preceding a variant (SV, indel) or at the point change (SNV).
#
#     :param df: Variant dataframe as BED.
#     :param ref_fa: Reference file.
#     """
#
#     # Get the reference base location
#     with pysam.FastaFile(ref_fa) as ref_file:
#
#         df = (
#             df.with_columns(
#                 (
#                     pl.when(pl.col('_ref_left'))
#                     .then(pl.col('pos') - 1)
#                     .otherwise(pl.col('end'))
#                 ).alias('_ref_base_loc')
#             )
#             .with_columns(
#                 pl.when(pl.col('_is_snv'))
#                 .then(pl.col('_ref_base_loc') + 1)
#                 .otherwise(pl.col('_ref_base_loc'))
#             ).alias('_ref_base_loc')
#             .with_columns(
#                 pl.struct(['chrom', '_ref_base_loc'])
#                 .apply(
#                     lambda vals: ref_file.fetch(vals['chrom'], vals['_ref_base_loc'], vals['_ref_base_loc'] + 1)
#                 ).alias('_ref_base')
#             )
#             .drop('_ref_base_Loc')
#         )
#
#     # Open and update records
#     with pysam.FastaFile(ref_fa) as ref_file:
#         for index, row in df.iterrows():
#
#             if row['SVTYPE'] in {'INS', 'DEL', 'INSDEL', 'DUP', 'INV'}:
#                 yield ref_file.fetch(row['#CHROM'], row['POS'] + (-1 if row['POS'] > 0 else 0), row['POS']).upper()
#
#             elif row['SVTYPE'] == 'SNV':
#                 if 'REF' in row:
#                     yield row['REF']
#                 else:
#                     yield ref_file.fetch(row['#CHROM'], row['POS'], row['POS'] + 1).upper()
#
#             else:
#                 raise RuntimeError('Unknown variant type: "{}" at index {}'.format(row['VARTYPE'], index))
