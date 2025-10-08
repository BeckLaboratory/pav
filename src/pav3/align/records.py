"""Alignment base operations."""

__all__ = [
    'check_record',
    'count_ops',
    'check_matched_bases'
]

import collections
from typing import Any

import Bio.Seq
import numpy as np
import polars as pl
import pysam

from ..seq import seq_len

from . import op


def check_record(
        row: dict[str, Any],
        df_qry_fai: pl.DataFrame
) -> None:
    """Check alignment DatFrame record for sanity.

    Sanity checks include:

    - Query and reference positions are in the right order
    - Query and reference end positions agree with the start positions and lengths of query- and reference-consuming
        alignment operations
    - Clipping appears strictly at the ends of records and in the right order
    - No negative positions
    - Query and reference lengths are valid

    :param row: Alignment table record.
    :param df_qry_fai: Query FAI table.

    :raises ValueError: If the alignment record fails sanity checks.
    """
    if 'chrom' in df_qry_fai.columns:
        df_qry_fai = df_qry_fai.rename({'chrom': 'qry_id'})

    try:
        op_count = count_ops(row)

    except ValueError as e:
        raise ValueError(
            (
                'Operation error for alignment index {align_index} '
                '(qry={qry_id}:{qry_pos}-{qry_end}, ref={chrom}:{pos}-{end}): {e}'
            ).format(**row, e=str(e))
        ) from e

    qry_len = seq_len(row['qry_id'], df_qry_fai)

    # Query and reference positions are in the right order
    if row['qry_pos'] >= row['qry_end']:
        raise ValueError(
            (
                'qry_pos >= qry_end ({qry_pos:,d} >= {qry_end:,d}): '
                'align_index={align_index}, qry={qry_id}:{qry_pos:,d}-{qry_end:,d}, ref={chrom}:{pos:,d}-{end:,d}'
            ).format(**row)
        )

    if row['pos'] >= row['end']:
        raise ValueError(
            (
                'pos >= end ({pos:,d} >= {end:,d}): '
                'align_index={align_index}, qry={qry_id}:{qry_pos:,d}-{qry_end:,d}, ref={chrom}:{pos:,d}-{end:,d}'
            ).format(**row)
        )

    # No negative positions
    if row['pos'] < 0:
        raise ValueError(
            (
                'pos < 0 ({pos:,d}): '
                'align_index={align_index}, qry={qry_id}:{qry_pos:,d}-{qry_end:,d}, ref={chrom}:{pos:,d}-{end:,d}'
            ).format(**row)
        )

    if row['qry_pos'] < 0:
        raise ValueError(
            (
                'qry_pos < 0 ({qry_pos:,d}): '
                'align_index={align_index}, qry={qry_id}:{qry_pos:,d}-{qry_end:,d}, ref={chrom}:{pos:,d}-{end:,d}'
            ).format(**row)
        )

    # pos and end agree with lengths
    if row['pos'] + op_count['ref_bp'] != row['end']:
        raise ValueError(
            (
                'end mismatch: pos + ref_bp != end ({pos:,d} + {ref_bp:,d} != {end:,d}): '
                'align_index={align_index}, qry={qry_id}:{qry_pos:,d}-{qry_end:,d}, ref={chrom}:{pos:,d}-{end:,d}'
            ).format(**row, ref_bp=op_count['ref_bp'])
        )

    # Query POS and END agree with length
    if row['qry_pos'] + op_count['qry_bp'] != row['qry_end']:
        raise ValueError(
            (
                'qry_end mismatch: qry_pos + qry_bp != qry_end ({qry_pos:,d} + {qry_bp:,d} != {qry_end:,d}): '
                'align_index={align_index}, qry={qry_id}:{qry_pos:,d}-{qry_end:,d}, ref={chrom}:{pos:,d}-{end:,d}'
            ).format(**row, qry_bp=op_count['qry_bp'])
        )

    # Query ends are not longer than query lengths
    if row['qry_end'] > qry_len:
        raise ValueError(
            (
                'qry_end out of range for query "{qry_id}": qry_end > query length ({qry_end:,d} > {qry_len:,d}) '
                'align_index={align_index}, qry={qry_id}:{qry_pos:,d}-{qry_end:,d}, ref={chrom}:{pos:,d}-{end:,d}'
            ).format(**row, qry_len=qry_len)
        )


def count_ops(
        row: dict[str, Any],
        allow_m: bool = False
) -> dict[str | int, int]:
    """Get total lengths of alignment operations by type.

    Returns a dictionary with a key for each operation code and the total lengths of all operations of that type.
    Operation codes are integers and strings (duplicated in the dictionary for convenience).

    Additional keys:
    * "ref_bp": Total number of reference bases traversed.
    * "qry_bp": Total number of query bases traversed including soft-clipped bases.
    * "clip_l": Total number of clipped bases on the left (upstream) side.
    * "clip_r": Total number of clipped bases on the right (downstream) side.

    Returns a tuple of:
    * ref_bp: Reference bases traversed by CIGAR operations.
    * qry_bp: Query bases traversed by CIGAR operations. Does not include clipped bases.
    * clip_l: Total number of clipped bases on the left (upstream) side.
    * clip_h_l: Hard-clipped bases on the left (upstream) side.
    * clip_s_l: Soft-clipped bases on the left (upstream) side.
    * clip_r: Total number of clipped bases on the right (downstream) side.
    * clip_h_r: Hard-clipped bases on the right (downstream) side.
    * clip_s_r: Soft-clipped bases on the right (downstream) side.

    :param row: Alignment record.
    :param allow_m: If True, allow "M" operations (aligned bases, match or mismatch). PAV does not allow M operations.

    :returns: Dictionary with keys for each operation code and the total lengths of all operations of that type.

    :raises ValueError: If the alignment contains only clipped bases.
    :raises ValueError: If errors in soft and hard clipping are found (i.e. not at ends or ordered correctly).
    :raises ValueError: If the alignment contains M operations and not `allow_m`.
    """
    # Count operations
    op_counter = collections.Counter()

    op_tuples = op.row_to_tuples(row)

    n_ops = len(op_tuples)

    for op_code, op_len in op_tuples:
        op_counter[op_code] += op_len

    op_count: dict[str | int, int] = {
        op.OP_CHAR[op_code]: op_counter[op_code] for op_code in op.OP_LIST
    } | {
        op_code: op_counter[op_code] for op_code in op.OP_LIST
    }

    # Count reference, query, and total clipped bases
    op_count['ref_bp'] = np.sum([op_counter[op_code] for op_code in op.CONSUMES_REF_ARR])
    op_count['qry_bp'] = np.sum([op_counter[op_code] for op_code in op.CONSUMES_QRY_ARR])
    op_count['clip'] = np.sum([op_counter[op_code] for op_code in op.CLIP_SET])

    # Check for M
    if not allow_m and op_count['M'] > 0:
        raise ValueError('Found M operations in alignment operations: PAV requires =/X operations for matched bases')

    # Clipping - left
    op_count['clip_h_l'] = 0
    op_count['clip_s_l'] = 0

    i_l = 0
    while i_l < n_ops and (op_code := op_tuples[i_l][0]) in op.CLIP_SET:
        if op_code == op.H:
            if i_l != 0:
                raise ValueError(f'Left-most hard-clip operations must be first (index={i_l})')

            op_count['clip_h_l'] += op_tuples[i_l][1]

        if op_code == op.S:
            if op_count['clip_s_l'] > 0:
                raise ValueError(f'Multiple soft-clipping operations on left (upstream) side (index={i_l})')

            op_count['clip_s_l'] += op_tuples[i_l][1]

        i_l += 1

    # Clipping - right
    op_count['clip_h_r'] = 0
    op_count['clip_s_r'] = 0

    i_r = n_ops - 1
    while i_r >= 0 and (op_code := op_tuples[i_r][0]) in op.CLIP_SET:
        if op_code == op.H:
            if i_r != n_ops - 1:
                raise ValueError(f'Right-most hard-clip operations must be last (index={i_r}, len={n_ops})')

            op_count['clip_h_r'] += op_tuples[i_r][1]

        if op_code == op.S:
            if op_count['clip_s_r'] > 0:
                raise ValueError(f'Multiple soft-clipping operations on right (downstream) side (index={i_r})')

            op_count['clip_s_r'] += op_tuples[i_r][1]

        i_r -= 1

    # Clipping - other
    if i_l > i_r:
        raise ValueError('Alignment record has only clipped bases')

    op_count['clip_l'] = op_count['clip_h_l'] + op_count['clip_s_l']
    op_count['clip_r'] = op_count['clip_h_r'] + op_count['clip_s_r']

    if op_count['clip_l'] + op_count['clip_r'] != op_count['clip']:
        raise ValueError(
            f'Clipped bases must be on ends: '
            f'clip_l({op_count["clip_l"]:,}) + clip_r({op_count["clip_r"]:,}) != clip({op_count["clip"]:,})'
        )

    return op_count


def check_matched_bases(
        row: dict[str, Any] | pl.DataFrame,
        ref_fa_filename: str,
        qry_fa_filename: str
) -> None:
    """Check aligned bases for agreement between reference and query sequences.

    Using an alignment record, traverse the alignment operations for aligned and matching bases, then check the
    reference and query sequences for agreement. This is intended to identify alignment logic errors in the library
    from alignment table parsing or alignment trimming operations. This should be used for testing, but can be
    skipped when run in production.

    The reference and query sequences for this alignment record are loaded from `ref_fa_filename` and `qry_fa_filename`.
    The alignment operations are traversed and all match (= operation) and mismatch (X operation) bases are checked.
    For match bases, the reference and query must match at that locus, and for mismatch bases, the reference and
    query must not match at that locus. Any violations of these rules will raise ValueError.

    If `row` is a DataFrame, then check all records.

    :param row: Alignment record or a table of alignment records.
    :param ref_fa_filename: Reference FASTA filename.
    :param qry_fa_filename: Query FASTA filename.

    :raises ValueError: If there are any bases aligned and matching (= operation), but do not match the reference and
        query sequences at the reference and query positions indicated by a record.
    :raises ValueError: If there are any bases aligned and not matching (X operation), but do match the reference and
        query sequences at the reference and query positions indicated by a record.
    """
    # Recurse through records if row is a DataFrame.
    if isinstance(row, pl.DataFrame):
        i = 0

        for df_row in row.rows(named=True):
            try:
                check_matched_bases(df_row, ref_fa_filename=ref_fa_filename, qry_fa_filename=qry_fa_filename)
            except ValueError as e:
                raise ValueError(f'Alignment record {i}: {e}') from e
            i += 1

        return

    # Check one row
    op_arr = op.row_to_arr(row)

    ref_pos = np.concatenate([
        [0],
        np.cumsum(np.where(np.isin(op_arr[:, 0], op.CONSUMES_REF_ARR), op_arr[:, 1], 0))[:-1]
    ])

    qry_pos = np.concatenate([
        [0],
        np.cumsum(np.where(np.isin(op_arr[:, 0], op.CONSUMES_QRY_ARR), op_arr[:, 1], 0))[:-1]
    ])

    with pysam.FastaFile(ref_fa_filename) as ref_fa:
        seq_ref = ref_fa.fetch(row['chrom'], row['pos'], row['end']).upper()

    with pysam.FastaFile(qry_fa_filename) as qry_fa:
        seq_qry = qry_fa.fetch(row['qry_id'], row['qry_pos'], row['qry_end']).upper()

    if row['is_rev']:
        seq_qry = str(Bio.Seq.Seq(seq_qry).reverse_complement()).upper()

    # Check matched bases
    for index in np.where(op_arr[:, 0] == op.EQ)[0]:

        if seq_ref[
            ref_pos[index]:(ref_pos[index] + op_arr[index, 1])
        ] != seq_qry[
            qry_pos[index]:(qry_pos[index] + op_arr[index, 1])
        ]:
            err_seq_ref = seq_ref[ref_pos[index]:(ref_pos[index] + op_arr[index, 1])]
            err_seq_qry = seq_qry[qry_pos[index]:(qry_pos[index] + op_arr[index, 1])]

            # Find position of the first mismatch
            mismatch_index = np.where(
                np.array(list(err_seq_ref)) !=
                np.array(list(err_seq_qry))
            )[0][0]

            raise ValueError(
                f'Found a mismatched base in match operation ("=") alignment operation {index}: '
                f'ref={seq_ref[mismatch_index]}, query={seq_qry[mismatch_index]}'
            )

    # Check mismatched bases
    for index in np.where(op_arr[:, 0] == op.X)[0]:
        if seq_ref[
            ref_pos[index]:ref_pos[index] + op_arr[index, 1]
        ] == seq_qry[
            qry_pos[index]:qry_pos[index] + op_arr[index, 1]
        ]:
            # Find position of the first match
            match_index = np.where(
                seq_ref[ref_pos[index]:ref_pos[index] + op_arr[index, 1]] !=
                seq_qry[qry_pos[index]:qry_pos[index] + op_arr[index, 1]]
            )[0][0]

            raise ValueError(
                f'Found a matching base in mismatch operation ("X") alignment operation {index}: '
                f'ref={seq_ref[match_index]}, query={seq_qry[match_index]}'
            )
