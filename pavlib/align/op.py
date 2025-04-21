"""
Constants and functions for working with alignment operations. Operations are encoded in CIGAR strings or represented
as tuples of (op_code: int, op_len: int).
"""

import numpy as np

# CIGAR operations
INT_STR_SET = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
CIGAR_OP_SET = {'M', 'I', 'D', 'N', 'S', 'H', 'P', '=', 'X'}

M = 0
I = 1
D = 2
N = 3
S = 4
H = 5
P = 6
EQ = 7
X = 8

CLIP_SET = {S, H}
ALIGN_SET = {M, EQ, X}
EQX_SET = {EQ, X}

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

OP_CHAR_FUNC = np.vectorize(OP_CHAR.get)

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

def cigar_as_array(
        cigar_str: str
) -> np.ndarray[int, int]:
    """
    Get a numpy array with two dimensions (dtype int). The first column is the operation codes, the second column is the
    operation lengths.

    :param cigar_str: CIGAR string.

    :return: Array of operation codes and lengths (dtype int).
    """

    pos = 0
    max_pos = len(cigar_str)

    op_tuples = list()

    while pos < max_pos:

        len_pos = pos

        while cigar_str[len_pos] in INT_STR_SET:
            len_pos += 1

        if len_pos == pos:
            raise RuntimeError(f'Missing length in CIGAR string at index {pos}')

        if cigar_str[len_pos] not in CIGAR_OP_SET:
            raise RuntimeError(f'Unknown CIGAR operation {cigar_str[pos]}')

        op_tuples.append(
            (OP_CODE[cigar_str[len_pos]], int(cigar_str[pos:len_pos]))
        )

        pos = len_pos + 1

    return np.array(op_tuples)

def to_cigar_string(
        op_arr: np.ndarray
):
    """
    Generate a CIGAR string from operation codes.

    :param op_arr: Array of operations (n x 2, op_code/op_len columns) or a list of tuples (op_code: int, op_len: int).

    :return: A CIGAR string.
    """

    return ''.join(
        np.char.add(
            op_arr[:, 1].astype(str),
            OP_CHAR_FUNC(op_arr[:, 0])
        )
    )
