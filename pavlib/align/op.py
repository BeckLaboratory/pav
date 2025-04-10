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

# def as_tuples(
#         record: pd.Series | str | list[tuple[int | str, int]]
# ) -> Iterable[tuple[int, int]]:
#     """
#     Get an iterator for operation tuples. Each tuple is (op-code: int, op-len: int).
#
#     :param record: Alignment record or a CIGAR string.
#
#     :return: Iterator of CIGAR operation tuples.
#     """
#
#     if isinstance(record, pd.Series):
#         if 'CIGAR' not in record.index:
#             raise RuntimeError('Missing "CIGAR" column in alignment record')
#
#         cigar = record['CIGAR']
#
#     elif isinstance(record, str):
#         cigar = record
#
#     elif isinstance(record, list):
#         if len(record) > 0 and isinstance(record[0], tuple) and len(record[0]) == 2 and isinstance(record[0][0], int):
#             # Avoid re-translating CIGAR operation
#             for cigar_len, cigar_op in record:
#                 yield cigar_op, cigar_len
#
#         # Ensure CIGAR operations are integers (tuples may contain strings or integers)
#         for cigar_len, cigar_op in record:
#             yield OP_CODE[cigar_op], cigar_len
#
#         return
#
#     else:
#         raise RuntimeError(f'Unknown record type: {type(record)}')
#
#     pos = 0
#     max_pos = len(cigar)
#
#     while pos < max_pos:
#
#         len_pos = pos
#
#         while cigar[len_pos] in INT_STR_SET:
#             len_pos += 1
#
#         if len_pos == pos:
#             raise RuntimeError('Missing length in CIGAR string for query {} alignment starting at {}:{}: CIGAR index {}'.format(
#                 record['QRY_ID'], record['#CHROM'], record['POS'], pos
#             ))
#
#         if cigar[len_pos] not in CIGAR_OP_SET:
#             raise RuntimeError('Unknown CIGAR operation for query {} alignment starting at {}:{}: CIGAR operation {}'.format(
#                 record['QRY_ID'], record['#CHROM'], record['POS'], cigar[pos]
#             ))
#
#         yield OP_CODE[cigar[len_pos]], int(cigar[pos:len_pos])
#
#         pos = len_pos + 1

# def apply_tuples(
#         df: pd.DataFrame | pd.Series | Iterable[list[tuple[int | str, int]]],
#         func: Callable[[list[tuple[int, int]]], Any]
# ) -> pd.Series | list:
#     """
#     Apply a function to a list of CIGAR operations.
#
#     :param df: DataFrame of alignment records including a "CIGAR" column, a Series (CIGAR column), or a list of
#         of CIGAR operation tuples for each alignment record (list of list of tuples(int, int)).
#     :param func: Function to apply to each list of CIGAR operation tuples.
#
#     :return: A Series object of values computed by `func` if `df` is a DataFrame or Series. Otherwise, a list of values
#         computed by `func`.
#     """
#
#
#     if isinstance(df, pd.DataFrame):
#         return df.apply(as_tuples, axis=1).apply(func)
#
#     if isinstance(df, pd.Series):
#         return df.apply(as_tuples).apply(func)
#
#     return [
#         func(list(as_tuples(record))) for record in df
#     ]

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
