"""Alignment operation codes and handling routines.

Each alignment contains an ordered list of operations, each defined by an operation code (match, mismatch, insertion,
deletion) and an operation length.

Outside of an alignment table, the operations are represented by an (N x 2) integer matrix where the first column
is operation codes and the second is operation lengths.

Representation inside a table is similar. The "align_ops" colun is a struct with "op_code" and "op_len" fields, each
containing an integer array of the same length.
"""

__all__ = [
    'INT_STR_SET',
    'CIGAR_OP_SET',
    'M', 'I', 'D', 'N', 'S', 'H', 'P', 'EQ', 'X',
    'OP_LIST',
    'CLIP_SET',
    'ALIGN_SET',
    'EQX_SET',
    'OP_CHAR',
    'OP_CHAR_FUNC',
    'OP_CODE',
    'CONSUMES_QRY_ARR',
    'CONSUMES_REF_ARR',
    'ADV_REF_ARR',
    'ADV_QRY_ARR',
    'VAR_ARR',
    'cigar_to_arr',
    'arr_to_cigar',
    'row_to_arr',
    'arr_to_row',
    'row_to_tuples',
    'clip_soft_to_hard',
    'op_arr_add_coords',
]

from typing import Any, Optional

import numpy as np


INT_STR_SET: frozenset[str] = frozenset({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', })
"""Set of valid integer character strings representing operation codes."""

CIGAR_OP_SET: frozenset[str] = frozenset({'M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X', })
"""Set of valid operation characters."""

M: int = 0
"""Match or mismatch operation code."""

I: int = 1  # noqa: E741
"""Insertion operation code."""

D: int = 2
"""Deletion operation code."""

N: int = 3
"""Skipped region operation code."""

S: int = 4
"""Soft clipping operation code."""

H: int = 5
"""Hard clipping operation code."""

P: int = 6
"""Padding operation code."""

EQ: int = 7
"""Sequence match operation code."""

X: int = 8
"""Sequence mismatch operation code."""

OP_LIST: tuple[int, ...] = (M, I, D, N, S, H, P, EQ, X, )
"""Operation codes."""

CLIP_SET: frozenset[int] = frozenset({S, H, })
"""Clipping operation codes (soft and hard clipping)."""

ALIGN_SET: frozenset[int] = frozenset({M, EQ, X, })
"""Alignment operation codes (match, sequence match, mismatch)."""

EQX_SET: frozenset[int] = frozenset({EQ, X})
"""Exact match/mismatch operation codes."""

OP_CHAR: dict[int, str] = {
    M: 'M',
    I: 'I',
    D: 'D',
    N: 'N',
    S: 'S',
    H: 'H',
    P: 'P',
    EQ: '=',
    X: 'X'
}
"""Mapping from operation codes to their character representations."""

OP_CHAR_FUNC = np.vectorize(lambda val: OP_CHAR.get(val, '?'))
"""numpy.vectorize: Vectorized function to convert operation codes to characters."""

OP_CODE: dict[str, int] = {
    'M': M,
    'I': I,
    'D': D,
    'N': N,
    'S': S,
    'H': H,
    'P': P,
    '=': EQ,
    'X': X
}
"""Mapping from CIGAR characters to operation codes."""

CONSUMES_QRY_ARR: np.typing.NDArray[np.integer] = np.array([M, I, S, EQ, X])
"""Array of operation codes that consume query bases."""

CONSUMES_REF_ARR: np.typing.NDArray[np.integer] = np.array([M, D, N, EQ, X])
"""Array of operation codes that consume reference bases."""

ADV_REF_ARR: np.typing.NDArray[np.integer] = np.array([M, EQ, X, D])
"""Array of operation codes that advance the reference position."""

ADV_QRY_ARR: np.typing.NDArray[np.integer] = np.array([M, EQ, X, I, S, H])
"""Array of operation codes that advance the query position."""

VAR_ARR: np.typing.NDArray[np.integer] = np.array([X, I, D])
"""Array of operation codes that introduce variation."""


def cigar_to_arr(
        cigar_str: str
) -> np.ndarray[np.int_]:
    """Convert a CIGAR string to an array.

    :param cigar_str: CIGAR string.

    :returns: Array of operation codes and lengths (dtype int).

    :raises ValueError: If the CIGAR string is not properly formatted.
    """
    pos = 0
    max_pos = len(cigar_str)

    op_tuples = list()

    while pos < max_pos:

        len_pos = pos

        while cigar_str[len_pos] in INT_STR_SET:
            len_pos += 1

        if len_pos == pos:
            raise ValueError(f'Missing length in CIGAR string at index {pos}')

        if cigar_str[len_pos] not in CIGAR_OP_SET:
            raise ValueError(f'Unknown CIGAR operation {cigar_str[len_pos]}')

        op_tuples.append(
            (OP_CODE[cigar_str[len_pos]], int(cigar_str[pos:len_pos]))
        )

        pos = len_pos + 1

    return np.array(op_tuples)


def arr_to_cigar(
        op_arr: np.ndarray
) -> str:
    """Generate a CIGAR string from operation codes.

    :param op_arr: Array of operations (n x 2, op_code/op_len columns).

    :returns: A CIGAR string.
    """
    return ''.join(
        np.char.add(
            op_arr[:, 1].astype(str),
            OP_CHAR_FUNC(op_arr[:, 0])
        )
    )


def row_to_arr(
        row: dict[str, Any],
        dtype: type = np.int64
) -> np.ndarray:
    """Transform alignment operations in a row to an operation array.

    :param row: Full alignment record or just the "align_ops" field in a record (both are acceptable).
    :param dtype: Numpy data type.

    :returns: Array of operations (n x 2, op_code/op_len columns).
    """
    if 'align_ops' in row:
        align_ops = row['align_ops']
        from_row = True
    else:
        align_ops = row
        from_row = False

    if missing_labels := {'op_code', 'op_len'} - set(align_ops.keys()):
        source = 'row' if from_row else 'align_ops record'
        raise ValueError(f'Missing keys in alignment record: {", ".join(sorted(missing_labels))} (source: {source})')

    return np.column_stack(
        (
            np.array(align_ops['op_code'], dtype=dtype),
            np.array(align_ops['op_len'], dtype=dtype)
        )
    )


def arr_to_row(
        op_arr: np.ndarray,
        row: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Transform an operation array to a dictionary with "op_code" and "op_len" keys (one array each) for rows.

    :param op_arr: Array of operations (n x 2, op_code/op_len columns).
    :param row: Full alignment record.

    :returns: Dictionary with "op_code" and "op_len" keys.
    """
    align_ops = {
        'op_code': [int(x) for x in op_arr[:, 0]],
        'op_len': [int(x) for x in op_arr[:, 1]]
    }

    if row is not None:
        row['align_ops'] = align_ops

    return align_ops


def row_to_tuples(
        row: dict[str, Any]
) -> list[tuple[int, int]]:
    """Transform a "align_ops" in a row (dict of "op_code" and "op_len") to a list of (op_code, op_len) tuples.

    :param row: Full alignment record or just the "align_ops" field in a record (both are acceptable).

    :returns: List of (op_code, op_len) tuples for each alignment operation.
    """
    if 'align_ops' in row:
        align_ops = row['align_ops']
        from_row = True
    else:
        align_ops = row
        from_row = False

    if missing_labels := {'op_code', 'op_len'} - set(align_ops.keys()):
        source = 'row' if from_row else 'align_ops record'
        raise ValueError(f'Missing keys in alignment record: {", ".join(sorted(missing_labels))} (source: {source})')

    return list(zip(align_ops['op_code'], align_ops['op_len']))


def clip_soft_to_hard(
        op_arr: np.ndarray
) -> np.ndarray:
    """Shift soft clipped bases to hard clipped bases.

    :param op_arr: Array of operations (n x 2, op_code/op_len columns).

    :returns: Array of operations (n x 2, op_code/op_len columns).

    :raises ValueError: If the alignment contains only clipped bases.
    """
    clip_l = 0
    clip_l_i = 0
    clip_r = 0
    clip_r_i = op_arr.shape[0]

    while clip_l_i < clip_r_i and op_arr[clip_l_i, 0] in CLIP_SET:
        clip_l += op_arr[clip_l_i, 1]
        clip_l_i += 1

    while clip_r_i > clip_l_i and op_arr[clip_r_i - 1, 0] in CLIP_SET:
        clip_r += op_arr[clip_r_i - 1, 1]
        clip_r_i -= 1

    if clip_r_i == clip_l_i:
        if op_arr.shape[0] > 0:
            raise ValueError('Alignment consists only of clipped bases')

        return op_arr

    if clip_l > 0:
        op_arr = np.append([(H, clip_l)], op_arr[clip_l_i:], axis=0)

    if clip_r > 0:
        op_arr = np.append(op_arr[:clip_r_i], [(H, clip_r)], axis=0)

    return op_arr


def op_arr_add_coords(
        op_arr: np.ndarray,
        pos_ref: int = 0,
        add_index: bool = True
) -> np.ndarray:
    """Add coordinate and index columns to an operation array.

    Columns:

        0: Operation code
        1: Operation length
        2: Reference position
        3: Query position
        4: Index (optional, first record is 0, second is 1, etc). Allows rows to be dropped while keeping a record of
           the operation index.

    :param op_arr: Array of operations (n x 2, op_code/op_len columns).
    :param pos_ref: First aligned reference base in the sequence. Note the query position is determined by following
        clipping and alignment operations.
    :param add_index: Add index column.

    :returns: Array of operations (n x 4 or n x 5) with columns described above.

    :raises ValueError: If the CIGAR string is invalid.
    """
    adv_ref_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], ADV_REF_ARR)
    adv_qry_arr = op_arr[:, 1] * np.isin(op_arr[:, 0], ADV_QRY_ARR)

    ref_pos_arr = np.cumsum(adv_ref_arr) - adv_ref_arr + pos_ref
    qry_pos_arr = np.cumsum(adv_qry_arr) - adv_qry_arr

    # Check for zero-length operations (no operation or bad CIGAR length)
    if np.any((adv_ref_arr + adv_qry_arr) == 0):
        no_op_arr = op_arr[(adv_ref_arr + adv_qry_arr == 0) & (op_arr[:, 1] > 0)]
        no_len_arr = op_arr[op_arr[:, 1] == 0]

        op_set = ', '.join(sorted(set(no_op_arr[:, 0].astype(str))))
        len_set = ', '.join(sorted(set(no_len_arr[:, 0].astype(str))))

        if op_set:
            raise ValueError(f'Unexpected operations in CIGAR string: operation code(s) "{op_set}"')

        if len_set:
            raise ValueError(f'Zero-length operations CIGAR string: operation code(s) "{len_set}"')

    if add_index:
        return np.concatenate([
            op_arr,
            np.expand_dims(ref_pos_arr, axis=1),
            np.expand_dims(qry_pos_arr, axis=1),
            np.expand_dims(np.arange(op_arr.shape[0]), axis=1)
        ], axis=1)
    else:
        return np.concatenate([
            op_arr,
            np.expand_dims(ref_pos_arr, axis=1),
            np.expand_dims(qry_pos_arr, axis=1)
        ], axis=1)
