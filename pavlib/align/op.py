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

CONSUMES_QRY_ARR = np.array([M, I, S, EQ, X])
CONSUMES_REF_ARR = np.array([M, D, N, EQ, X])

def cigar_as_array(
        cigar_str: str
) -> np.ndarray:
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

    :param op_arr: Array of operations (n x 2, op_code/op_len columns).

    :return: A CIGAR string.
    """

    return ''.join(
        np.char.add(
            op_arr[:, 1].astype(str),
            OP_CHAR_FUNC(op_arr[:, 0])
        )
    )

def clip_soft_to_hard(
        op_arr: np.ndarray
) -> np.ndarray:
    """
    Shift soft clipped bases to hard clipped bases.

    :param op_arr: Array of operations (n x 2, op_code/op_len columns).

    :return: Array of operations (n x 2, op_code/op_len columns).
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
            raise RuntimeError(f'Alignment consists only of clipped bases')

        return op_arr

    if clip_l > 0:
        op_arr = np.append([(H, clip_l)], op_arr[clip_l_i:], axis=0)

    if clip_r > 0:
        op_arr = np.append(op_arr[:clip_r_i], [(H, clip_r)], axis=0)

    return op_arr
