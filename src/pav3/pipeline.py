"""PAV pipeline utilities.

Functions for finding data.
"""

__all__ = [
    'expand_pattern',
    'get_hap_list',
    'get_asm_config',
    'get_asm_input_list',
    'expand_input',
    'get_rule_input_list',
    'input_tuples_to_fasta',
    'get_override_config',
    'read_assembly_table',
    'expand_fofn',
    'link_fasta',
]

import itertools
import os
from pathlib import Path
from typing import Any, Iterable, Optional
import re
import pysam

import Bio.SeqIO
import Bio.bgzf
import polars as pl

import agglovar

from .params import PavParams
from .seq import fa_to_record_iter, gfa_to_record_iter


def expand_pattern(
        pattern: str,
        asm_table: pl.DataFrame,
        pav_config: PavParams,
        **kwargs
) -> Iterable[str]:
    """Get a list of files matching a pattern with wildcards.

    Wildcards are embedded in the pattern string (surrounded by curly braces). Standard PAV pipeline wildcard values are
    recognized, such as "asm_name", "hap", and "trim". A list of string (likely filenames) is returned with one item
    for each valid combination of wildcard values.

    :param pattern: String pattern with wildcards.
    :param asm_table: Assembly table.
    :param pav_config: Dictionary of PAV configuration values.

    :yields: Pattern strings with wildcards expanded.
    """
    processed_kwargs: dict[str, tuple[Any, ...]] = {}

    # Check kwargs
    if kwargs is None or len(kwargs) == 0:
        # Empty kwargs so the kwargs product iterates over one item
        processed_kwargs = {'': (None,)}

    for key, val in kwargs.items():
        if isinstance(val, str):
            # Single string value, iterate over one item, not each string character
            processed_kwargs[key] = (val,)
        else:
            try:
                processed_kwargs[key] = tuple(val)
            except TypeError:
                processed_kwargs[key] = (val,)

    kwargs_keys = [val for val in sorted(processed_kwargs.keys()) if val not in {'asm_name', 'hap'}]
    kwargs_n = len(kwargs_keys)

    sub_dict = dict()  # String substitution dict

    if 'asm_name' in processed_kwargs.keys() and processed_kwargs['asm_name'] is not None:
        asm_list = set(processed_kwargs['asm_name'])
    else:
        asm_list = set(asm_table.select('name').to_series())

    if 'hap' in processed_kwargs.keys() and processed_kwargs['hap'] is not None:
        hap_set = set(processed_kwargs['hap'])
    else:
        hap_set = None

    if pav_config is not None:
        if 'asm_name' in pav_config:
            asm_list &= {val.strip() for val in pav_config['asm_name'].split(',') if val.strip()}

        if 'hap' in pav_config:
            config_haps = {val.strip() for val in pav_config['hap'].split(',') if val.strip()}

            hap_set = (hap_set & config_haps) if hap_set is not None else config_haps

    # Process each assembly
    for asm_name in asm_list:
        sub_dict['asm_name'] = asm_name

        for hap in get_hap_list(asm_name, asm_table):

            if hap_set is not None and hap not in hap_set:
                continue

            sub_dict['hap'] = hap

            for kw_prod in itertools.product(*[processed_kwargs[key] for key in kwargs_keys]):
                for i in range(kwargs_n):
                    sub_dict[kwargs_keys[i]] = kw_prod[i]

                yield pattern.format(**sub_dict)


def get_hap_list(
        asm_name: str,
        asm_table: pl.DataFrame
) -> list[str]:
    """Get a list of haplotypes for an assembly.

    :param asm_name: Assembly name.
    :param asm_table: Assembly table.

    :returns: List of haplotypes.

    :raises ValueError: If the named assembly is not in the assembly table.
    :raises ValueError: if no haplotypes were found for the assembly.
    """
    # Check values
    if asm_name is None or (asm_name := asm_name.strip()) == '':
        raise RuntimeError('Cannot get assembly config: "asm_name" is missing')

    # Get assembly table entry
    if asm_name not in asm_table['name']:
        raise RuntimeError(f'Assembly name not found in the sample config: {asm_name}')

    asm_table_entry = asm_table.row(by_predicate=pl.col('name') == asm_name, named=True)

    # Find haplotypes for a table entry
    hap_list = [
        col[len('hap_'):]
        for col in asm_table_entry.keys()
        if col.startswith('hap_') and asm_table_entry[col] is not None
    ]

    if len(hap_list) == 0:
        raise ValueError(f'No haplotypes found for assembly {asm_name}: All hap columns are missing or empty')

    return hap_list


def get_asm_config(
        asm_name: str,
        hap: str,
        asm_table: pl.DataFrame
) -> dict[str, Any]:
    """Get a dictionary of parameters and paths for one assembly.

    :param asm_name: Assembly name.
    :param hap: Haplotype name (e.g. "h1", "h2").
    :param asm_table: Assembly table.

    :returns: A dictionary of parameters and paths.

    :raises ValueError: If the named assembly or haplotype is not in the assembly table.
    """
    # Check values
    if hap is None or (hap := hap.strip()) == '':
        raise ValueError('Cannot get assembly config: "hap" is missing')

    if asm_name is None or (asm_name := asm_name.strip()) == '':
        raise ValueError('Cannot get assembly config: "asm_name" is missing')

    if asm_name not in asm_table['name']:
        raise ValueError(f'No assembly table entry: {asm_name}')

    if f'hap_{hap}' not in asm_table.columns:
        raise ValueError(f'No haplotype in assembly table columns: {hap}')

    # Get assembly table entry
    asm_table_entry = asm_table.row(by_predicate=pl.col('name') == asm_name, named=True)

    # Get filename pattern
    assembly_input = asm_table_entry[f'hap_{hap}']

    if assembly_input is not None:
        assembly_input = assembly_input.format(asm_name=asm_name, sample=asm_name, hap=hap)
        assembly_input = [val.strip() for val in assembly_input.split(';') if val.strip()]
    else:
        assembly_input = []

    # Return config dictionary
    return {
        'asm_name': asm_name,
        'hap': hap,
        'assembly_input': assembly_input,
    }


def get_asm_input_list(
        asm_name: str,
        hap: str,
        asm_table: pl.DataFrame
) -> list[str]:
    """Get a list of input files.

    :param asm_name: Assembly name.
    :param hap: Haplotype.
    :param asm_table: Assembly table.

    :returns: List of input files.

    :raises FileNotFoundError: If any of the input files are missing, empty, or not regular files.
    """
    # Get config
    assembly_input = get_asm_config(asm_name, hap, asm_table)['assembly_input']

    empty_assembly = [
        file_name for file_name in assembly_input
        if not os.path.isfile(file_name) or os.stat(file_name).st_size == 0
    ]

    if empty_assembly:
        raise FileNotFoundError(
            f'Found {len(empty_assembly)} input file name(s) that ar missing, empty, or not regular files: '
            f'asm_name={asm_name}, hap={hap}: {", ".join(empty_assembly)}'
        )

    return assembly_input


def expand_input(
        file_name_list: list[str]
) -> tuple[list[tuple[str, str]], list[str]]:
    """Expand input to a list of tuples containing the file name and type.

    Tuple elements:

        - 0: File name.
        - 1: File type ("fasta", "fastq", or "gfa")

    File type does not change if the file is gzipped (i.e. ".fasta" and ".fasta.gz" are both type "fasta").

    This function traverses FOFN files (file of file names) recursively until FASTA, FASTQ, or GFA files are found.

    Returns A tuple of two lists:

        - A list of tuples containing the file name and type.
        - A list of FOFN file names.


    :param file_name_list: List of input file names.

    :returns: A tuple of two lists.

    :raises ValueError: If file types cannot be determined from file names.
    """
    # Check arguments
    if file_name_list is None:
        raise RuntimeError('Cannot create input FASTA: Input name list is None')

    # Check files
    if isinstance(file_name_list, str):
        file_name_list = [file_name_list]
    else:
        try:
            file_name_list = list(file_name_list)
        except TypeError as e:
            raise ValueError(f'Error expanding input: Input name list is not iterable: {type(file_name_list)}') from e

    # Generate a list of files traversing into FOFN files
    file_name_tuples = []
    fofn_list = []  # Set of visited FOFN files (prevent recursive traversal)

    while len(file_name_list) > 0:

        # Next file name
        file_name = file_name_list[0]
        file_name_list = file_name_list[1:]

        if not (file_name := file_name.strip()):
            continue

        # Get file extension
        file_name_lower = file_name.lower()

        if file_name_lower.endswith('.gz'):  # Strip GZ, the downstream input functions will detect file type
            file_name_lower = file_name_lower.rsplit('.', 1)[0]

        if '.' not in file_name_lower:
            raise ValueError(f'No recognizable extension in file name: {file_name}')

        file_name_ext = file_name_lower.rsplit('.', 1)[1]

        # Expand FOFN files
        if file_name_ext == 'fofn':

            # Check for recursive FOFN traversal
            file_name_real = os.path.realpath(file_name)

            if file_name_real in fofn_list:
                raise ValueError(f'Detected recursive FOFN traversal, ignoring redundant entry: {file_name}')

            fofn_list.append(file_name_real)

            # Append FOFN entries to the input file list
            with agglovar.io.PlainOrGzReader(file_name) as in_file:
                for line in in_file:
                    line = line.strip()

                    if line:
                        file_name_list.append(line)

        elif file_name_ext in {'fasta', 'fa', 'fn', 'fna'}:
            file_name_tuples.append((file_name, 'fasta'))

        elif file_name_ext in {'fastq', 'fq', 'fnq'}:
            file_name_tuples.append((file_name, 'fastq'))

        elif file_name_ext == 'gfa':
            file_name_tuples.append((file_name, 'gfa'))

        else:
            raise ValueError(f'Unrecognized file extension {file_name_ext}: {file_name}')

    # Return tuples
    return file_name_tuples, fofn_list


def get_rule_input_list(
        asm_name: str,
        hap: str,
        asm_table: pl.DataFrame
) -> list[str]:
    """Get a full list of input files.

    :param asm_name: Assembly name.
    :param hap: Haplotype.
    :param asm_table: Assembly table.

    :returns: A list of all files that may affect input. This includes both sequence data files and FOFN files.
    """
    input_list = get_asm_input_list(asm_name, hap, asm_table)

    file_name_tuples, fofn_list = expand_input(input_list)

    return [
        filename for filename in fofn_list
    ] + [
        filename for filename, filetype in file_name_tuples
    ]


def input_tuples_to_fasta(
        file_name_tuples: Iterable[tuple[str, str]],
        out_file_name: str
) -> None:
    """Convert a list of input files to a single FASTA entry.

    Input files may be FASTA, FASTQ, or GFA.

    :param file_name_tuples: List of tuples for each input entry ([0]: File name, [1]: File format). The file format
        must be "fasta", "fastq", or "gfa" (case sensitive).
    :param out_file_name: Output file to write gzipped FASTA data.

    :raises ValueError: If there are no input files.
    :raises ValueError: If output file name is missing.
    :raises ValueError: If file types cannot be determined.
    :raises RuntimeError: If any of the input files are missing, empty, or not regular files.
    """
    if out_file_name is None or (out_file_name := out_file_name.strip()) == '':
        raise ValueError('Cannot create input FASTA from sources: Output file name is empty')

    # Check input files, fail early
    if file_name_tuples is None or not file_name_tuples:
        raise ValueError('Cannot create input FASTA from sources: No input files')

    for file_name, file_format in file_name_tuples:
        if file_format not in {'fasta', 'fastq', 'gfa'}:
            raise ValueError(
                f'Cannot create input FASTA from sources: Unrecognized file format "{file_format}": {file_name}'
            )

        if not os.path.isfile(file_name):
            raise FileNotFoundError(
                f'Cannot create input FASTA from sources: Input file does not exist or is not a regular file: '
                f'{file_name}'
            )

        if os.stat(file_name).st_size == 0:
            raise ValueError(f'Cannot create input FASTA from sources: Input file is empty: {file_name}')

    # Record iterator
    def input_record_iter():
        record_id_set = set()

        for file_name, file_format in file_name_tuples:

            if file_format in {'fasta', 'fastq'}:
                for record in fa_to_record_iter(file_name, input_format=file_format):

                    if record.id in record_id_set:
                        raise ValueError(f'Duplicate record ID in input: {record.id}')

                    record_id_set.add(record.id)

                    yield record

            elif file_format == 'gfa':
                for record in gfa_to_record_iter(file_name):

                    if record.id in record_id_set:
                        raise ValueError(f'Duplicate record ID in input: {record.id}')

                    record_id_set.add(record.id)

                    yield record

            elif file_format not in {'skip', 'empty'}:
                raise ValueError(f'Unrecognized file type "{file_format}" after checking input.')

    # Write all records
    with Bio.bgzf.open(out_file_name, 'wb') as out_file:
        Bio.SeqIO.write(input_record_iter(), out_file, 'fasta')


def _get_config_override_dict(config_string: str) -> dict[str, Any]:
    """Get a dictionary of overridden parameters using the "config" column of the assembly table.

    :param config_string: Config override string (e.g. attr1=val1;attr2=val2). Must be colon separated and each
        element must have an equal sign. Whitespace around semi-colons and equal signs is ignored.

    :returns: Dict of overridden parameters or an empty dict if no parameters were overridden.

    :raises ValueError: If config string cannot be parsed into a set of attribute-value pairs.
    """
    config_override = dict()

    # Check string
    if config_string is None or not (config_string := config_string.strip()):
        return config_override

    # Process each config directive
    tok_list = config_string.split(';')

    for tok in tok_list:

        # Check tok
        tok = tok.strip()

        if not tok:
            continue

        if '=' not in tok:
            raise ValueError(f'Cannot get assembly config: Missing "=" in CONFIG token {tok}: {config_string}')

        # Get attribute and value
        key, val = tok.split('=', 1)

        key = key.strip()
        val = val.strip()

        if not key:
            raise ValueError(
                f'Cannot get assembly config: Missing key (key=value) in CONFIG token {tok}: {config_string}'
            )

        if not val:
            raise ValueError(
                f'Cannot get assembly config: Missing value (key=value) in CONFIG token {tok}: {config_string}'
            )

        # Set
        config_override[key] = val

    return config_override


def _get_config_with_override(
        config: dict[str, Any],
        override_config: dict[str, Any]
) -> dict[str, Any]:
    """Get a config dict with values replaced by overridden values.

    The dict in parameter `config` is copied if it is modified. The original (unmodified) config or a modified copy is
    returned.

    :param config: Existing config. Original object will not be modified.
    :param override_config: A defined set of values that will override entries in `config`.

    :returns: A config object.

    :raises ValueError: If the reference configuration parameter is defined per sample. References must be across
        all samples.
    """
    if override_config is None:
        return config

    if config is None:
        config = dict()

    config = config.copy()

    for key, val in override_config.items():
        if key in {'reference'}:
            raise ValueError('The reference configuration parameter cannot be defined per sample.')

        config[key] = val

    return config


def get_override_config(
        pav_config: dict[str, Any],
        asm_name: str,
        asm_table: pl.DataFrame
) -> dict[str, Any]:
    """Get a config dict with values replaced by overridden values.

    The dict in parameter `pav_config` is copied if it is modified. The original (unmodified) `pav_config` or a
    modified copy is returned.

    :param pav_config: Existing PAV config. Original object will not be modified.
    :param asm_name: Name of the assembly.
    :param asm_table: Assembly table.

    :returns: A dictionary of PAV configuration parameters with overrides f rom the assembly table applied.

    :raises RowsError: If the assembly name is not present or not unique in the assembly table.
    """
    if asm_name is None:
        raise ValueError('Cannot get override for assembly: None')

    if asm_table is None:
        raise ValueError('Cannot get override for assembly table: None')

    # Get table entry
    if asm_name not in asm_table['name']:
        return pav_config

    asm_table_entry = asm_table.row(by_predicate=pl.col('name') == asm_name, named=True)

    if 'config' not in asm_table_entry or asm_table_entry['config'] is None:
        return pav_config

    return _get_config_with_override(
        pav_config,
        _get_config_override_dict(asm_table_entry['config'])
    )


def read_assembly_table(
        filename: str,
        pav_config: Optional[dict[str, Any]] = None
) -> pl.DataFrame:
    """Read assembly table.

    The table returned by this function renames the input columns (e.g. "HAP1" or "HAP_h1") to the haplotype name (e.g.
    "h1") and keeps them in the order they were found in the input table. All other columns are removed.

    The fields returned are:

        - name: Assembly name.
        - hap_NAME: Assembly source for a haplotype with NAME (e.g. NAME == "h1").
        - config: Assembly configuration (e.g. "reference" or "sample" or "reference,sample").

    The columns are normalized such that empty haplotype or configuration fields contain Null (not empty strings).
    There will always be a "name" column, a "config" column, and at least one haplotype column. All strings in these
    standard columns are stripped of leading and trailing whitespace.

    :param filename: Input filename to read. If None, produce an empty table.
    :param pav_config: Pipeline configuration.

    :returns: Assembly table.

    :raises ValueError: If the table cannot be read.
    :raises ValueError: The table contains errors, such as duplicate or empty assembly names.
    :raises ValueError: If the table contains no haplotypes.
    :raises FileNotFoundError: If the table file does not exist or is not a regular file.
    """
    if filename is None:
        raise ValueError('Cannot read assembly table: None')

    filename = filename.strip()

    if not os.path.isfile(filename):
        raise FileNotFoundError(f'Assembly table file missing or is not a regular file: {filename}')

    if pav_config is None:
        pav_config = dict()

    ignore_cols = set(pav_config.get('ignore_cols', set())) | {'config', 'name'}

    # Read table
    filename_lower = filename.lower()

    if filename_lower.endswith(('.tsv', '.tsv.gz', '.tsv.txt', 'tsv.txt.gz')):
        df = pl.read_csv(filename, separator='\t', infer_schema_length=0)
    elif filename_lower.endswith('.xlsx'):
        df = pl.read_excel(filename, infer_schema_length=0)
    elif filename_lower.endswith(('.csv', '.csv.gz', '.csv.txt', '.csv.txt.gz')):
        df = pl.read_csv(filename, infer_schema_length=0)
    elif filename_lower.endswith('parquet'):
        df = pl.read_parquet(filename)
    else:
        raise ValueError(
            f'Unrecognized table file type (expected ".tsv", ".tsv.gz", ".xlsx", ".csv", ".csv.gz", or ".parquet"): '
            f'{filename}'
        )

    # Make standard columns lowercase
    col_map = dict()

    for col in df.columns:
        col_lower = col.lower()

        if col_lower not in {'name', 'config'}:
            continue

        if col_lower in col_map:
            raise ValueError(f'Duplicate column names found in assembly table when ignoring case: {filename}')

        col_map[col] = col_lower

    df = df.rename(col_map)

    # Check/set standard columns
    if 'name' not in df.columns:
        raise ValueError('Missing assembly table column: name')

    if 'config' not in df.columns:
        df = df.with_columns(pl.lit(None).alias('config').cast(pl.String))

    # Check assembly names
    df = (
        df
        .with_columns(pl.col('name').str.strip_chars().alias('name'))
        .with_columns(
            pl.when(pl.col('name') == '')
            .then(None)
            .otherwise(pl.col('name'))
            .alias('name')
        )
    )

    if df.select(pl.col('name').is_null().any())[0, 0]:
        raise ValueError(f'Found null entries in the assembly table with empty name values: {filename}')

    if df.select(pl.col('name').n_unique())[0, 0] < df.height:
        raise ValueError(f'Found duplicate name values in the assembly table: {filename}')

    bad_name = {name for name in df['name'] if '\t' in name}

    if bad_name:
        raise ValueError(f'Found {len(bad_name)} column names with tab characters in the name: {filename}')

    # Map haplotype names to column names
    hap_list = list()
    hap_col_map = dict()
    filter_list = list()
    unknown_cols = list()

    for col in set(df.columns) - ignore_cols:

        if match_hap_named := re.search(r'^HAP_([a-zA-Z0-9-+.]+)$', col, re.IGNORECASE):
            hap = match_hap_named[1]

        elif match_hap_num := re.search(r'^HAP([0-9]+)$', col, re.IGNORECASE):
            hap = f'h{match_hap_num[1]}'

        elif re.search(r'^FILTER_([a-zA-Z0-9-+.]+)$', col, re.IGNORECASE):
            filter_list.append(col)
            continue

        else:
            unknown_cols.append(col)
            continue

        hap_list.append(hap)

        if hap in hap_col_map:
            if col != hap_col_map[hap]:
                dup_source = f'(duplicate column {col})'
            else:
                dup_source = f'(derived from columns {hap_col_map[hap]} and {col})'

            raise ValueError(f'Duplicate haplotype name "{hap}" found in assembly table {dup_source}: {filename}')

        hap_col_map[hap] = col

    if unknown_cols:
        col_list = ', '.join(unknown_cols[:5]) + '...' if len(unknown_cols) > 5 else ''
        raise ValueError(f'Unknown columns in assembly table: {col_list}: {filename}')

    if not hap_list:
        raise ValueError(f'No haplotype columns found in assembly table: {filename}')

    # Set index and column names
    df = df.rename({col: f'hap_{hap}' for hap, col in hap_col_map.items()})

    # Strip hap and config columns, set empty values to null
    for hap in hap_list:
        df = (
            df
            .with_columns(
                pl.col(f'hap_{hap}')
                .str.strip_chars()
            )
            .with_columns(
                pl.when(
                    pl.col(f'hap_{hap}') == ''
                )
                .then(None)
                .otherwise(pl.col(f'hap_{hap}'))
                .alias(f'hap_{hap}')
            )
        )

    df = (
        df
        .with_columns(
            pl.col('config')
            .str.strip_chars()
        )
        .with_columns(
            pl.when(
                pl.col('config') == ''
            )
            .then(None)
            .otherwise(pl.col('config'))
            .alias('config')
        )
    )

    return df


def expand_fofn(
        fofn_filename: str | Path,
        include_fofn: bool = False,
        recurse: bool = False,
        fail_not_found: bool = True,
        comment: Optional[str] = '#',
        rel: Optional[str | Path] = '.'
) -> list[Path]:
    """Expand an FOFN file to a list of filenames it contains.

    :param fofn_filename: FOFN file name.
    :param include_fofn: Include FOFN file names in the output list. If True, then both the top-level FOFN is included
        and all FOFN files recursed if recurse is True.
    :param recurse: Recurse into FOFN files if True.
    :param fail_not_found: Raise an error if a file within the FOFN is not found. If the top-level FOFN file is not
        found, an error is always raised.
    :param comment: Ignore lines starting with this character (leading/trailing whitespace is stripped before
        comparison). If None, accept all lines.
    :param rel: Relative path to the FOFN file. If None, files in the FOFN are relative to the FOFN's parent directory.

    :returns: List of file names.
    """
    fofn_filename = Path(fofn_filename)

    if not fofn_filename.is_file():
        raise FileNotFoundError(f'File not found or not a regular file: {str(fofn_filename)}')

    comment = comment if comment is not None else ' '  # Whitespace is stripped, so this disables comments

    rel = Path(rel) if rel is not None else None

    fofn_in_list: list[Path] = [fofn_filename]
    out_list = []

    while fofn_in_list:
        fofn_path = fofn_in_list.pop()

        if include_fofn:
            out_list.append(fofn_path)

        if not fofn_path.is_file():
            if fail_not_found:
                raise FileNotFoundError(f'FOFN file not found or not a regular file: {str(fofn_path)}')

            continue

        # Read FOFN
        with agglovar.io.PlainOrGzReader(fofn_path) as in_file:
            for line in in_file:
                line = line.strip()

                if not line or line.startswith(comment):
                    continue

                fofn_rel = rel if rel is not None else fofn_path.parent

                file_path = fofn_rel / Path(line)

                if fail_not_found and not file_path.is_file():
                    raise FileNotFoundError(f'File inside FOFN not found or not a regular file: {str(file_path)}')

                file_path_is_fofn = recurse and re.search(r'\.fofn(\.gz)?$', line, re.IGNORECASE) is not None

                if file_path_is_fofn:
                    fofn_in_list.append(file_path)

                    if include_fofn:
                        out_list.append(file_path)
                else:
                    out_list.append(file_path)

    return out_list


def link_fasta(
        source: str | Path,
        dest: str | Path
) -> list[Path]:
    """Link a FASTA file and indexes to a location.

    The destination filename should include the FASTA extension (e.g., ".fa" or ".fasta") and may include ".gz". If
    it includes ".gz", but the source file does not, then an error is raised. If ".gz" is not in the destination
    filename, it is added if the source file ends with ".gz"..

    :param source: Input FASTA file.
    :param dest: Output basename.

    :returns: A list of paths to files created.
    """
    # Check arguments
    source = str(source).strip()
    dest = str(dest).strip()

    source_is_gz = str(source).lower().endswith('.gz')
    dest_is_gz = str(dest).lower().endswith('.gz')

    if dest_is_gz and not source_is_gz:
        raise ValueError(f'Cannot link gzipped FASTA file to non-gzipped FASTA file: {source} -> {dest}')

    if not dest_is_gz and source_is_gz:
        dest += '.gz'
        dest_is_gz = True

    source_fai = source + '.fai'
    source_gzi = source + '.gzi'

    dest_fai = dest + '.fai'
    dest_gzi = dest + '.gzi'

    # Check source
    if not os.path.isfile(source):
        raise FileNotFoundError(f'Input FASTA file is missing or not a regular file: {source}')

    source_is_indexed = os.path.isfile(source_fai) and (
        os.path.isfile(source_gzi) if source_is_gz else True
    )

    dest_fa_path = Path(dest)
    dest_fai_path = Path(dest_fai)
    dest_gzi_path = Path(dest_gzi) if dest_is_gz else None

    # Link FASTA
    dest_fa_path.unlink(missing_ok=True)
    dest_fa_path.symlink_to(source)

    # Link indexes (if present)
    if source_is_indexed:
        dest_fai_path.unlink(missing_ok=True)
        dest_fai_path.symlink_to(source_fai)

        if dest_gzi_path is not None:
            dest_gzi_path.unlink(missing_ok=True)
            dest_gzi_path.symlink_to(source_gzi)

    else:
        # Create indexes
        pysam.faidx(str(dest_fa_path))

    # Return a file list
    return [
        dest_fa_path, dest_fai_path
    ] + [dest_gzi_path] if dest_gzi_path is not None else []
