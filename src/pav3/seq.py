"""Routines for aligned contigs."""

__all__ = [
    'ref_kmers',
    'region_seq_fasta',
    'variant_seq_from_region',
    'seq_len',
    'fa_to_record_iter',
    'gfa_to_record_iter',
]

import collections
import os
from typing import Iterable, Iterator, Optional, Self

import agglovar
import Bio.Seq
import Bio.SeqIO
import Bio.SeqRecord
import polars as pl
import pysam

from .region import Region


def ref_kmers(
        seq_region: Region,
        fa_file_name: str,
        k_util: agglovar.kmer.util.KmerUtil
) -> collections.Counter:
    """Get a counter keyed by k-mers.

    :param seq_region: Region to extract.
    :param fa_file_name: FASTA file to extract sequence from.
    :param k_util: K-mer utility for k-merizing sequence.

    :returns: A collections.Counter object key k-mer keys and counts.
    """
    ref_seq = region_seq_fasta(seq_region, fa_file_name, False)

    ref_mer_count = collections.Counter()

    for kmer in agglovar.kmer.util.stream(ref_seq, k_util):
        ref_mer_count[kmer] += 1

    return ref_mer_count


def region_seq_fasta(
        seq_region: Region,
        fa_file_name: str,
        rev_compl: Optional[bool] = None
) -> str:
    """Get sequence from an indexed FASTA file. FASTA must have ".fai" index.

    :param seq_region: Region object to extract a region, or a string with the record ID to extract a whole record.
    :param fa_file_name: FASTA file name.
    :param rev_compl: Reverse-complement sequence is `True`. If `None`, reverse-complement if `region.is_rev`.

    :returns: String sequence.
    """
    with pysam.FastaFile(fa_file_name) as fa_file:

        if isinstance(seq_region, str):
            is_region = False
        elif isinstance(seq_region, Region):
            is_region = True
        else:
            raise ValueError(f'Unrecognized region type: {type(seq_region)}: Expected Region or str')

        if is_region:
            sequence = fa_file.fetch(seq_region.chrom, seq_region.pos, seq_region.end)
        else:
            sequence = fa_file.fetch(seq_region, None, None)

        if rev_compl is None:
            if is_region and seq_region.is_rev:
                return str(Bio.Seq.Seq(sequence).reverse_complement())
        else:
            if rev_compl:
                return str(Bio.Seq.Seq(sequence).reverse_complement())

        return sequence


def variant_seq_from_region(
        df: pl.DataFrame,
        fa_filename: str,
        region_col: Iterable[str] = ('qry_id', 'qry_pos', 'qry_end'),
        strand_col: str = 'qry_strand',
        id_col: str = 'id',
        seq_upper: bool = False
):
    """Get sequence from an indexed FASTA file. FASTA must have ".fai" index.

    :param df: Variant DataFrame.
    :param fa_filename: FASTA file name.
    :param region_col: Region column names. Columns must be in order: 1) Chromosome or query name, 2) Start position,
        3) End position.
    :param strand_col: Strand column name. If column is "-" or `True`, the sequence is reverse complemented. Must be
        "+", "-", `True`, or `False`. The boolean values allow "is_rev" column to be used where the sequence is on
        the negative strand if `True`. If `None`, no reverse complementing is performed.
    :param id_col: ID column name.
    :param seq_upper: Upper-case sequences if `True` (removes soft-mask annotations in the sequence string).

    :returns: Iterator over Bio.Seq sequence objects with sequences in reference orientation.

    :raises ValueError: If FASTA file name is missing.
    :raises ValueError: If required columns are not found in the variant table.
    :raises FileNotFoundError: If FASTA file does not exist or is not a regular file.
    """
    raise NotImplementedError
    #
    # if df is None:
    #     raise RuntimeError('Variant DataFrame is missing')
    #
    # fa_file_name = fa_filename.strip() if fa_filename is not None else None
    #
    # if fa_file_name is None or len(fa_file_name) == 0:
    #     raise ValueError('FASTA file name is missing')
    #
    # if not os.path.isfile(fa_file_name):
    #     raise FileNotFoundError(f'FASTA file does not exist or is not a regular file: {fa_file_name}')
    #
    # if strand_col is not None and strand_col not in df.columns:
    #     raise ValueError(f'Strand column not found in DataFrame: "{strand_col}"')
    #
    # if id_col not in  df.columns:
    #     raise ValueError(f'ID column not found in DataFrame: "{id_col}"')
    #
    # n_blank = np.sum(df[id_col].apply(len) == 0)
    #
    # if n_blank > 0:
    #     raise ValueError(f'Found {n_blank} empty IDs in DataFrame: "{id_col}"')
    #
    # dup_ids = sorted([id for id, count in collections.Counter(df[id_col]).items() if count > 1])
    #
    # if dup_ids:
    #     n_dup = len(dup_ids)
    #     dup_ids = ', '.join(dup_ids[3:]) + (', ...' if n_dup > 3 else '')
    #     raise ValueError(f'Found {n_dup} duplicate IDs in DataFrame: "{id_col}": {dup_ids}')
    #
    # # Get regions
    # if isinstance(region_col, str):
    #     if region_col not in df.columns:
    #         raise ValueError(f'Region column not found in DataFrame: "{region_col}"')
    #
    #     df_region = df[region_col].apply(region.region_from_string)
    #
    # else:
    #     region_col = list(region_col)
    #
    #     if len(region_col) != 3:
    #         raise ValueError(f'Expected 3 columns for region column: "{region_col}"')
    #
    #     df_region = df[region_col].apply(lambda row: region.Region(*row), axis=1)
    #
    # else:
    #     raise ValueError(f'Unrecognized region column type: "{region_col}"')
    #
    # # Extract sequences
    # with io.FastaReader(fa_file_name) as fa_file:
    #
    #     for region, qry_strand, var_id in zip(df_region, df[strand_col], df[id_col]):
    #
    #         seq = Bio.Seq.Seq(fa_file.fetch(region.chrom, region.pos, region.end))
    #
    #         if seq_upper:
    #             seq = seq.upper()
    #
    #         if strand_col is not None:
    #             if qry_strand in {'-', True}:
    #                 seq = seq.reverse_complement()
    #             elif qry_strand not in {'+', False}:
    #                 raise RuntimeError(f'Unrecognized strand in record {row[id_col]}: "{row[strand_col]}"')
    #
    #         yield Bio.SeqRecord.SeqRecord(
    #             seq, id=var_id, description='', name=''
    #         )


def seq_len(
        name: str,
        df_fai: pl.DataFrame,
        col_chrom: Optional[str] = None,
        col_len: str = 'len'
) -> int:
    """Get the length of a sequence from a FAI table.

    :param name: Name of the sequence.
    :param df_fai: DataFrame of FAI file.
    :param col_chrom: Column name for the sequence name (typically "chrom" or "qry_id").
    :param col_len: Column name for the sequence length.

    :returns: Sequence length.

    :raises ValueError: If the columns are not found in the FAI table.
    :raises ValueError: If the sequence is not found in the FAI table.
    """
    if col_chrom is None:
        col_list = list(set(df_fai.columns) & {'chrom', 'qry_id'})

        if len(col_list) == 0:
            raise ValueError('No "chrom" or "qry_id" in DataFrame')

        if len(col_list) > 1:
            raise ValueError('Found both "chrom" and "qry_id" in DataFrame')

        col_chrom = col_list[0]
    else:
        if col_chrom not in df_fai.columns:
            raise ValueError(f'Column not found in DataFrame: "{col_chrom}"')

    if col_len not in df_fai.columns:
        raise ValueError(f'Length column not found in DataFrame: "{col_len}"')

    try:
        return int(
            df_fai.row(
                by_predicate=pl.col(col_chrom) == name,
                named=True
            )[col_len]
        )

    except pl.NoRowsReturnedError:
        raise ValueError(f'Sequence not found in FAI table: "{name}"')


def fa_to_record_iter(
        fa_file_name: str,
        record_set: Optional[set[str] | dict[str, str]] = None,
        require_all: bool = True,
        input_format: str = 'fasta'
) -> Iterator[Bio.SeqRecord.SeqRecord]:
    """Get an iterator for records in a FASTA file.

    Returns an iterator of SeqIO.Seq objects.

    The optional `record_set` parameter can be used to filter and/or rename records. If `record_set` is a `set`, only
    extract records with IDs in this set. If `record_set` is a `dict`, rename record IDs (keys: original ID, values:
    new ID) and only extract records with IDs in this dict.

    :param fa_file_name: FASTA file name. May be gzipped or plain text.
    :param record_set: Set of record names to extract or dict for mapping records. If `None`, extract all records.
    :param require_all: If `True`, raise an error if all records from record_set are not found.
    :param input_format: Input file format. Must be "fasta" or "fastq" (case sensitive).

    :yields: Bio.SeqRecord.SeqRecord objects for each sequence record.

    :raises ValueError: If `record_set` is not a set or dict or is empty.
    :raises ValueError: If `input_format` is not "fasta" or "fastq".
    :raises ValueError: If multiple records with the same ID are found.
    :raises KeyError: If all records from `record_set` are not found.
    """
    # Check file format
    if input_format not in {'fasta', 'fastq'}:
        raise ValueError(f'Unrecognized input format: "{input_format}"')

    # Setup record_set and record_dict for translating record names
    if record_set is not None:
        if isinstance(record_set, dict):
            record_dict = record_set
            record_set = set(record_dict.keys())

        elif isinstance(record_set, set):
            record_dict = {val: val for val in record_set}

        else:
            raise ValueError(f'Parameter record_set must be a set or a dict: {record_set.__class__}')

        if len(record_dict) == 0:
            raise ValueError('Parameter record_set must not be empty')

    else:
        record_set = None
        record_dict = None

    # Open and parse records
    found_set = set()

    with agglovar.io.PlainOrGzReader(fa_file_name) as in_file:
        for record in Bio.SeqIO.parse(in_file, input_format):

            record_id_org = record.id

            # Translate and/or filter records
            if record_dict is not None:
                record_id = record_dict.get(record.id, None)

                if record_id is None:
                    continue

                record.id = record_id

            # Check for duplicate records
            if record.id in found_set:
                raise ValueError('Duplicate {} entries for record: {}{}'.format(
                    input_format.upper(),
                    record.id,
                    f' (original ID {record_id_org})' if record.id != record_id_org else ''
                ))

            # Yield record
            found_set.add(record.id)
            yield record

    # Check for required records
    if require_all and record_set is not None and found_set != record_set:
        missing_set = record_set - found_set

        raise KeyError('Missing {} records when parsing {} {}: {}{}'.format(
            len(missing_set),
            input_format.upper(),
            fa_file_name,
            ', '.join(sorted(missing_set)[:3]),
            '...' if len(missing_set) > 3 else ''
        ))


def gfa_to_record_iter(
        gfa_file_name: str,
        record_set: Optional[set[str] | dict[str, str]] = None,
        require_all: bool = True,
):
    """Open a GFA file and parse "S" lines into sequence records.

    The optional `record_set` parameter can be used to filter and/or rename records. If `record_set` is a `set`, only
    extract records with IDs in this set. If `record_set` is a `dict`, rename record IDs (keys: original ID, values:
    new ID) and only extract records with IDs in this dict.

    :param gfa_file_name: GFA file name. May be gzipped or plain text.
    :param record_set: Set of record names to extract or dict for mapping records. If `None`, extract all records.
    :param require_all: If `True`, raise an error if all records from record_set are not found.

    :yields: Bio.SeqRecord.SeqRecord objects for each sequence record.

    :raises ValueError: If `record_set` is not a set or dict or is empty.
    :raises ValueError: If multiple records with the same ID are found.
    :raises KeyError: If all records from `record_set` are not found.
    """
    # Setup record_set and record_dict for translating record names
    if record_set is not None:
        if isinstance(record_set, dict):
            record_dict = record_set
            record_set = set(record_dict.keys())

        elif isinstance(record_set, set):
            record_dict = {val: val for val in record_set}

        else:
            raise ValueError(f'Parameter record_set must be a set or a dict: {record_set.__class__}')

        if len(record_dict) == 0:
            raise ValueError('Parameter record_set must not be empty')

    else:
        record_set = None
        record_dict = None

    # Open and parse records
    found_set = set()

    # Open and parse records
    with agglovar.io.PlainOrGzReader(gfa_file_name) as in_file:

        line_count = 0

        for line in in_file:
            line_count += 1

            # Get record ID and sequence
            tok = line.split('\t')

            if tok[0] != 'S':
                continue

            if len(tok) < 3:
                raise ValueError(
                    f'Error reading GFA "S" record at line {line_count}: '
                    f'Expected at least 3 tab-separated columns: {gfa_file_name}'
                )

            record_id = tok[1].strip()
            record_seq = tok[2].strip()

            if not record_seq:
                continue

            # Check for duplicate IDs
            if record_id in found_set:
                raise ValueError('Duplicate GFA entries for record: {}'.format(record_id))

            found_set.add(record_id)

            # Translate record names
            if record_dict is not None:
                record_id = record_dict.get(record_id, record_id)

            # Yield record
            yield Bio.SeqRecord.SeqRecord(
                Bio.Seq.Seq(record_seq),
                id=record_id,
                description=''
            )

    # Check for required records
    if require_all and record_set is not None and found_set != record_set:
        missing_set = record_set - found_set

        raise KeyError('Missing {} records when parsing GFA {}: {}{}'.format(
            len(missing_set),
            gfa_file_name,
            ', '.join(sorted(missing_set)[:3]),
            '...' if len(missing_set) > 3 else ''
        ))


class LRUSequenceCache:
    """Cache for sequence records from an indexed FASTA file.

    Caches the last `max_size` records and retrieves new sequences from the indexed FASTA file if they are not cached.
    """

    def __init__(
            self,
            fa_filename: str,
            max_size: int,
            upper: bool = False
    ) -> None:
        """Init a cache object.

        :param fa_filename: Indexed FASTA file name.
        :param max_size: Maximum number of records to cache.
        :param upper: If True, sequences are made upper-case.

        :raises ValueError: If `fa_filename` is missing or empty.
        :raises FileNotFoundError: If `fa_filename` does not exist or is not a regular file.
        """
        if fa_filename is None or (fa_filename := str(fa_filename).strip()) == '':
            raise ValueError('FASTA file name is missing or empty')

        if not os.path.isfile(fa_filename):
            raise FileNotFoundError(f'FASTA file does not exist or is not a regular file: {fa_filename}')

        self._fa_filename = fa_filename
        self._fa_file = None
        self._upper = upper

        self._cache = collections.OrderedDict()
        self._max_size = max_size
        self._open_context = False

    @property
    def fa_filename(self) -> str:
        """FASTA file name."""
        return self._fa_filename

    @property
    def upper(self) -> bool:
        """If True, sequences are made upper-case."""
        return self._upper

    @property
    def fa_file(self) -> Optional[pysam.FastaFile]:
        """pysam.FastaFile object for indexed FASTA file or None if file is not open."""
        return self._fa_file

    @property
    def max_size(self) -> int:
        """Maximum number of records to cache."""
        return self._max_size

    @property
    def is_open(self) -> bool:
        """If True, the indexed FASTA file is open."""
        return self._fa_file is not None

    def open(self) -> None:
        """Open file.

        Not recommended to call directly, use as context manager (with LRUSequenceCache() as seq_cache: ...).

        :raises ValueError: If FASTA file is already open.
        """
        if self._fa_file is not None:
            raise ValueError('FASTA file is already open')

        self._fa_file = pysam.FastaFile(self._fa_filename)

    def close(self):
        """Close file.

        Not recommended to call directly, use as context manager
        (with LRUSequenceCache() as seq_cache: ...).

        :raises ValueError: If FASTA file was opened as a context manager.
        """
        if self._open_context:
            raise ValueError('FASTA file is open as context manager')

        if self._fa_file is not None:
            self._fa_file.close()
            self._fa_file = None

    def __getitem__(
            self,
            key: str | tuple[str, bool]
    ) -> str:
        """
        Get a sequence record from the FASTA file.

        :param key: Sequence record ID or a tuple of the record ID and a boolean flag indicating if the sequence should
            be reverse-complemented (True) or not (False). If tuples are used, then the cache will maintain forward
            and reverse-complemented sequences as separate records.

        :returns: Sequence.

        :raises ValueError: If FASTA file is not open.
        :raises KeyError: If FASTA file does not contain a sequence for the key.
        """
        if isinstance(key, str):
            key = (key, False)

        elif not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError(f'Invalid key: {key}: Expected str or (str, bool)')

        if self._fa_file is None:
            raise ValueError('FASTA file is not open: Use as context manager or call open()')

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if (key[0], not key[1]) in self._cache:
            seq = str(Bio.Seq.Seq(self._cache[(key[0], not key[1])]).reverse_complement())

            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = seq

            return seq

        try:
            seq = self._fa_file.fetch(key[0]).strip()

            if self._upper:
                seq = seq.upper()

            if key[1]:
                seq = str(Bio.Seq.Seq(seq).reverse_complement())

            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = seq

            return seq

        except KeyError:
            raise KeyError(f'FASTA file {self._fa_filename} does not contain a sequence for key: {key}')

    def is_cached(
            self,
            key: str
    ) -> bool:
        """Check if a sequence record is cached.

        :param key: Sequence record ID.

        :returns: True if the sequence record is cached, False otherwise (sequence pulled from the FASTA file).
        """
        return key in self._cache

    def clear(
            self,
            key: Optional[str] = None
    ) -> None:
        """Clear the cache or remove a record.

        If a `key` is specified, remove it from the cache if it exists. If it does not exist, do nothing (no error).

        :param key: Sequence record ID to clear. If None, clear all records.
        """
        if key is not None:
            if key in self._cache:
                del self._cache[key]
        else:
            self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Get the cache size."""
        return len(self._cache)

    def __enter__(self) -> Self:
        """Enter context manager."""
        self.open()
        self._open_context = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit context manager."""
        if self._fa_file is not None:
            self.clear()
            self._fa_file.close()
            self._fa_file = None
            self._open_context = False
