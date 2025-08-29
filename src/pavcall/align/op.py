"""
Constants and functions for working with alignment operations. Operations are encoded in CIGAR strings or represented
as tuples of (op_code: int, op_len: int).
"""

from typing import Any, Optional

import numpy as np
import polars as pl

# Character sets for parsing CIGAR strings
INT_STR_SET = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
"""set[str]: Set of valid integer character strings for CIGAR parsing."""

CIGAR_OP_SET = {'M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X'}
"""set[str]: Set of valid CIGAR operation characters."""

# CIGAR operation codes (following SAM specification)
M = 0
"""int: Match or mismatch operation code."""

I = 1
"""int: Insertion operation code."""

D = 2
"""int: Deletion operation code."""

N = 3
"""int: Skipped region operation code."""

S = 4
"""int: Soft clipping operation code."""

H = 5
"""int: Hard clipping operation code."""

P = 6
"""int: Padding operation code."""

EQ = 7
"""int: Sequence match operation code."""

X = 8
"""int: Sequence mismatch operation code."""

OP_LIST = [M, I, D, N, S, H, P, EQ, X]
"""list[int]: List of valid CIGAR operation codes."""

# Operation sets for categorization
CLIP_SET = {S, H}
"""set[int]: Set of clipping operation codes (soft and hard clipping)."""

ALIGN_SET = {M, EQ, X}
"""set[int]: Set of alignment operation codes (match, sequence match, mismatch)."""

EQX_SET = {EQ, X}
"""set[int]: Set of exact match/mismatch operation codes."""

# Mapping dictionaries
OP_CHAR = {
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
"""dict[int, str]: Mapping from operation codes to CIGAR characters."""

OP_CHAR_FUNC = np.vectorize(lambda val: OP_CHAR.get(val, '?'))
"""numpy.vectorize: Vectorized function to convert operation codes to characters."""

OP_CODE = {
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
"""dict[str, int]: Mapping from CIGAR characters to operation codes."""

CONSUMES_QRY_ARR = np.array([M, I, S, EQ, X])
"""numpy.ndarray: Array of operation codes that consume query bases."""

CONSUMES_REF_ARR = np.array([M, D, N, EQ, X])
"""numpy.ndarray: Array of operation codes that consume reference bases."""

ADV_REF_ARR = np.array([M, EQ, X, D])
"""numpy.ndarray: Array of operation codes that advance the reference position."""

ADV_QRY_ARR = np.array([M, EQ, X, I, S, H])
"""numpy.ndarray: Array of operation codes that advance the query position."""

VAR_ARR = np.array([X, I, D])
"""numpy.ndarray: Array of operation codes that introduce variation."""


def cigar_to_arr(
        cigar_str: str
) -> np.ndarray:
    """
    Get a numpy array with two dimensions (dtype int). The first column is the operation codes, the second column is the
    operation lengths.

    Args:
        cigar_str: CIGAR string.

    Returns:
        Array of operation codes and lengths (dtype int).

    Raises:
        ValueError: If the CIGAR string is invalid.
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
    """
    Generate a CIGAR string from operation codes.

    Args:
        op_arr: Array of operations (n x 2, op_code/op_len columns).

    Returns:
        A CIGAR string.
    """

    return ''.join(
        np.char.add(
            op_arr[:, 1].astype(str),
            OP_CHAR_FUNC(op_arr[:, 0])
        )
    )

def arr_to_tuples(
        op_arr: np.ndarray
) -> list[tuple[int, int]]:
    """
    Convert an operation array to a list of (op_code, op_len) tuples.

    Args:
        op_arr: Array of operations (n x 2, op_code/op_len columns).

    Returns:
        List of (op_code, op_len) tuples.
    """

    return list(zip(
        [int(x) for x in op_arr[:, 0]],
        [int(x) for x in op_arr[:, 1]]
    ))

def tuples_to_arr(
        op_tuples: list[tuple[int, int]],
        dtype: type = np.int64
) -> np.ndarray:
    """
    Convert a list of (op_code, op_len) tuples to an operation array.

    Args:
        op_tuples: List of (op_code, op_len) tuples.
        dtype: Numpy data type.

    Returns:
        Array of operations (n x 2, op_code/op_len columns).
    """
    return np.array(op_tuples, dtype=dtype)

def row_to_arr(
        row: dict[str, Any],
        dtype: type = np.int64
) -> np.ndarray:
    """
    Transform alignment operations in a row to an operation array.

    Args:
        row: Full alignment record or just the "align_ops" field in a record (both are acceptable).
        dtype: Numpy data type.

    Returns:
        Array of operations (n x 2, op_code/op_len columns).
    """

    if 'align_ops' in row:
        align_ops = row['align_ops']
        from_row =True
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
    """
    Transform an operation array to a dictionary with "op_code" and "op_len" keys (one array each) for rows.

    Args:
        op_arr: Array of operations (n x 2, op_code/op_len columns).
        row: Full alignment record.

    Returns:
        Dictionary with "op_code" and "op_len" keys.
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
    """
    Transform a "align_ops" in a row (dict of "op_code" and "op_len") to a list of (op_code, op_len) tuples.

    Args:
        row: Full alignment record or just the "align_ops" field in a record (both are acceptable).

    Returns:
        List of (op_code, op_len) tuples for each alignment operation.
    """

    if 'align_ops' in row:
        align_ops = row['align_ops']
        from_row =True
    else:
        align_ops = row
        from_row = False

    if missing_labels := {'op_code', 'op_len'} - set(align_ops.keys()):
        source = 'row' if from_row else 'align_ops record'
        raise ValueError(f'Missing keys in alignment record: {", ".join(sorted(missing_labels))} (source: {source})')

    return list(zip(align_ops['op_code'], align_ops['op_len']))

def tuples_to_row(
        op_tuples: list[tuple[int, int]],
        row: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Transform a list of (op_code, op_len) tuples to a dictionary with "op_code" and "op_len" keys (one array each).

    Args:
        op_tuples: List of (op_code, op_len) tuples.
        row: Alignment record. If set, "align_ops" will be added to the record. The "align_ops" dictionary will still
            be returned by the function.

    Returns:
        Dictionary with "op_code" and "op_len" keys.
    """

    align_ops = {
        'op_code': [int(op[0]) for op in op_tuples],
        'op_len': [int(op[1]) for op in op_tuples]
    }

    if row is not None:
        row['align_ops'] = align_ops

    return align_ops

def clip_soft_to_hard(
        op_arr: np.ndarray
) -> np.ndarray:
    """
    Shift soft clipped bases to hard clipped bases.

    Args:
        op_arr: Array of operations (n x 2, op_code/op_len columns).

    Returns:
        Array of operations (n x 2, op_code/op_len columns).

    Raises:
        ValueError: If the alignment contains only clipped bases.
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
            raise ValueError(f'Alignment consists only of clipped bases')

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
    """
    Take an operation array (n x 2, op_code/op_len columns), add coordinate and index columns.

    Columns:
        0: Operation code
        1: Operation length
        2: Reference position
        3: Query position
        4: Index (optional, first record is 0, second is 1, etc). Allows rows to be dropped while keeping a record of
            the operation index.

    Args:
        op_arr: Array of operations (n x 2, op_code/op_len columns).
        pos_ref: First aligned reference base in the sequence. Note the query position is determined by following
            clipping and alignment operations.
        add_index: Add index column.

    Returns:
        Array of operations (n x 4 or n x 5) with columns described above.

    Raises:
        ValueError: If the CIGAR string is invalid.
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
