# Alignment base operations


import numpy as np
import pandas as pd

from . import op


# Filter definitions
FILTER_LCALIGN = 'LCALIGN'

# def count_aligned_bases(
#         record: pd.Series | str | list[tuple[str, int]]
# ) -> int:
#     """
#     Count the number of bases aligned in an alignment record ignoring bases in gaps. This is an appropriate denominator
#     for calculating the fraction of mismatch bases over aligned bases.
#     """
#     return sum([cigar_len for cigar_op, cigar_len in op.as_tuples(record) if cigar_op in op.ALIGN_SET])

# def get_mismatch_prop(
#         record: pd.Series | str | list[tuple[str, int]]
# ) -> float:
#     """
#     Get the proportion of mismatches in an alignment record defined as the number of mismatches (CIGAR "X") divided by
#     the number of aligned bases (CIGAR "=" and CIGAR "X"). Gaps are ignored. CIGAR "M' operations throw ValueError.
#
#     :param record: Alignment record or a CIGAR string. If the type is a pandas Series, the "CIGAR" column is used. If
#         it is a string, then it is assumed to be a CIGAR string. Otherwise, it is assumed to be a list of cigar
#         operations (i.e. from as_cigar_tuples()).
#
#     :return: Mismatch proportion (float) or `np.nan` if there are no aligned bases.
#     """
#
#     match_bp = 0
#     mismatch_bp = 0
#
#     for cigar_op, cigar_len in op.as_tuples(record):
#         if cigar_op == op.EQ:
#             match_bp += cigar_len
#
#         elif cigar_op == op.X:
#             mismatch_bp += cigar_len
#
#         elif cigar_op == op.M:
#             raise ValueError('Cannot compute mismatch prop for CIGAR "M" operations')
#
#     align_bp = match_bp + mismatch_bp
#
#     if align_bp == 0:
#         return np.nan
#
#     return mismatch_bp / align_bp

# def get_match_bp(
#         record: pd.Series | str | list[tuple[str, int]],
#         right_end: bool
# ) -> int:
#     """
#     Get the number of matching bases at the end of an alignment. Used by variant callers to left-align SVs through
#     alignment-truncating events.
#
#     :param record: Alignment record (from alignment BED) with CIGAR string.
#     :param right_end: `True` if matching alignments from the right end of `record`, or `False` to match from
#         the left end.
#
#     :return: Minimum of the number of matched bases at the end of two alignment records.
#     """
#
#     cigar = list(op.as_tuples(record))
#
#     if right_end:
#         cigar = cigar[::-1]
#
#     # Get match base count (CIGAR op "=") on a
#     match_count = 0
#
#     for cigar_op, cigar_len in cigar:
#         if cigar_op in op.CLIP_SET:  # Skip clipped bases: S, H
#             continue
#
#         if cigar_op == op.EQ:  # Matched bases: =
#             match_count += cigar_len
#
#         elif cigar_op == op.M:
#             raise RuntimeError(
#                 'Detected "M" opcodes in CIGAR string for record INDEX={}: Sequence match/mismatch opcodes are required ("=", "X")'.format(
#                     record['INDEX'] if 'INDEX' in record.index else '<UNKNOWN>'
#                 )
#             )
#         else:
#             break  # No more bases to traverse
#
#     return match_count


def check_record(
        row: pd.Series,
        df_qry_fai: pd.Series
) -> None:
    """
    Check alignment DatFrame record for sanity. Throws exceptions if there are problems. Returns nothing if everything
    passes.

    :param row: Alignment table record (Pandas Series).
    :param df_qry_fai: Panadas Series with query names as keys and query lengths as values.
    """

    try:
        ref_bp, qry_bp, clip_h_l, clip_s_l, clip_h_r, clip_s_r = count_cigar(row)

    except Exception as ex:
        raise RuntimeError(
            'CIGAR parsing error: {} (INDEX={}, QRY={}:{}-{}, REF={}:{}-{})'.format(
                ex,
                row['INDEX'], row['QRY_ID'], row['QRY_POS'], row['QRY_END'], row['#CHROM'], row['POS'], row['END']
            )
        )

    qry_len = df_qry_fai[row['QRY_ID']]

    # Query and reference positions are in the right order
    if row['QRY_POS'] >= row['QRY_END']:
        raise RuntimeError('QRY_POS >= QRY_END ({} >= {}) (INDEX={}, QRY={}:{}-{}, REF={}:{}-{})'.format(
            row['QRY_POS'], row['QRY_END'],
            row['INDEX'], row['QRY_ID'], row['QRY_POS'], row['QRY_END'], row['#CHROM'], row['POS'], row['END']
        ))

    if row['POS'] >= row['END']:
        raise RuntimeError('POS >= END ({} >= {}) (INDEX={}, QRY={}:{}-{}, REF={}:{}-{})'.format(
            row['POS'], row['END'],
            row['INDEX'], row['QRY_ID'], row['QRY_POS'], row['QRY_END'], row['#CHROM'], row['POS'], row['END']
        ))

    # No negative positions
    if row['POS'] < 0:
        raise RuntimeError('POS ({}) < 0 (INDEX={}, QRY={}:{}-{}, REF={}:{}-{})'.format(
            row['POS'],
            row['INDEX'], row['QRY_ID'], row['QRY_POS'], row['QRY_END'], row['#CHROM'], row['POS'], row['END']
        ))

    if row['QRY_POS'] < 0:
        raise RuntimeError('QRY_POS ({}) < 0 (INDEX={}, QRY={}:{}-{}, REF={}:{}-{})'.format(
            row['QRY_POS'],
            row['INDEX'], row['QRY_ID'], row['QRY_POS'], row['QRY_END'], row['#CHROM'], row['POS'], row['END']
        ))

    # if qry_map_rgn.pos < 0:
    #     raise RuntimeError('QRY_MAP_RGN.pos ({}) < 0 (INDEX={}, QRY={}:{}-{}, REF={}:{}-{})'.format(
    #         qry_map_rgn.pos,
    #         row['INDEX'], row['QRY_ID'], row['QRY_POS'], row['QRY_END'], row['#CHROM'], row['POS'], row['END']
    #     ))

    # POS and END agree with length
    if row['POS'] + ref_bp != row['END']:

        raise RuntimeError(
            'END mismatch: POS + ref_bp != END ({} != {}) (INDEX={}, QRY={}:{}-{}, REF={}:{}-{})'.format(
                row['POS'] + ref_bp, row['END'],
                row['INDEX'], row['QRY_ID'], row['QRY_POS'], row['QRY_END'], row['#CHROM'], row['POS'], row['END']
            )
        )

    # Query POS and END agree with length
    if row['QRY_POS'] + qry_bp != row['QRY_END']:
        raise RuntimeError(
            'QRY_POS + qry_bp != QRY_END: {} != {} (INDEX={}, QRY={}:{}-{}, REF={}:{}-{})'.format(
                row['QRY_POS'] + qry_bp, row['QRY_END'],
                row['INDEX'], row['QRY_ID'], row['QRY_POS'], row['QRY_END'], row['#CHROM'], row['POS'], row['END']
            )
        )

    # Query ends are not longer than query lengths
    if row['QRY_END'] > qry_len:
        raise RuntimeError('QRY_END > qry_len ({} > {}) (INDEX={}, QRY={}:{}-{}, REF={}:{}-{})'.format(
            row['QRY_END'], qry_len,
            row['INDEX'], row['QRY_ID'], row['QRY_POS'], row['QRY_END'], row['#CHROM'], row['POS'], row['END']
        ))


def check_record_err_string(
        df: pd.DataFrame,
        df_qry_fai: pd.Series
):
    """
    Runs check_record on each row of `df`, captures exceptions, and returns a Series of error message strings instead
    of failing on the first error. The Series can be added as a column to `df`. For each record where there was no
    error, the field for that record in the returned series is NA (`np.nan`). This function may not be used by the
    pipeline, but is here for troubleshooting alignments.

    :param df: Dataframe of alignment records.
    :param df_qry_fai: Panadas Series with query names as keys and query lengths as values.

    :return: A Series of error messages (or NA) for each record in `df`.
    """

    def _check_record_err_string_check_row(row):
        try:
            check_record(row, df_qry_fai)
            return None
        except Exception as ex:
            return str(ex)

    return df.apply(_check_record_err_string_check_row, axis=1)


def count_cigar(
        record: pd.Series,
        allow_m: bool=False
) -> tuple[int, int, int, int, int, int]:
    """
    Count bases affected by CIGAR operations in an alignment record (row is a Pandas Series from an ailgnment BED).

    Returns a tuple of:
    * ref_bp: Reference bases traversed by CIGAR operations.
    * qry_bp: Query bases traversed by CIGAR operations. Does not include clipped bases.
    * clip_h_l: Hard-clipped bases on the left (upstream) side.
    * clip_s_l: Soft-clipped bases on the left (upstream) side.
    * clip_h_r: Hard-clipped bases on the right (downstream) side.
    * clip_s_r: Soft-clipped bases on the right (downstream) side.

    :param record: Row with CIGAR records as a CIGAR string.
    :param allow_m: If True, allow "M" CIGAR operations. PAV does not allow M operations, this option exists for other
        tools using the PAV library.

    :return: A tuple of (ref_bp, qry_bp, clip_h_l, clip_s_l, clip_h_r, clip_s_r).
    """

    ref_bp = 0
    qry_bp = 0

    clip_s_l = 0
    clip_h_l = 0

    clip_s_r = 0
    clip_h_r = 0

    op_arr = op.cigar_as_array(record['CIGAR'])

    op_n = op_arr.shape[0]

    index = 0

    while index < op_n and op_arr[index, 0] in op.CLIP_SET:

        if op_arr[index, 0] == op.S:
            if clip_s_l > 0:
                raise RuntimeError('Duplicate S records (left) at index {}'.format(index))
            clip_s_l = op_arr[index, 1]

        if op_arr[index, 0] == op.H:
            if clip_h_l > 0:
                raise RuntimeError('Duplicate H records (left) at index {}'.format(index))

            if clip_s_l > 0:
                raise RuntimeError('S record before H (left) at index {}'.format(index))

            clip_h_l = op_arr[index, 1]

        index += 1

    while index < op_n:

        if op_arr[index, 0] in op.EQX_SET:

            if clip_s_r > 0 or clip_h_r > 0:
                raise RuntimeError(
                    'Found clipped bases before last non-clipped CIGAR operation at operation {} ({}{})'.format(
                        index, op_arr[index, 1], op_arr[index, 0]
                    )
                )

            ref_bp += op_arr[index, 1]
            qry_bp += op_arr[index, 1]

        elif op_arr[index, 0] == op.I:

            if clip_s_r > 0 or clip_h_r > 0:
                raise RuntimeError(
                    'Found clipped bases before last non-clipped CIGAR operation at operation {} ({}{})'.format(
                        index, op_arr[index, 1], op_arr[index, 0]
                    )
                )

            qry_bp += op_arr[index, 1]

        elif op_arr[index, 0] == op.D:

            if clip_s_r > 0 or clip_h_r > 0:
                raise RuntimeError(
                    'Found clipped bases before last non-clipped CIGAR operation at operation {} ({}{})'.format(
                        index, op_arr[index, 1], op_arr[index, 0]
                    )
                )

            ref_bp += op_arr[index, 1]

        elif op_arr[index, 0] == op.S:

            if clip_s_r > 0:
                raise RuntimeError('Duplicate S records (right) at operation {}'.format(index))

            if clip_h_r > 0:
                raise RuntimeError('H record before S record (right) at operation {}'.format(index))

            clip_s_r = op_arr[index, 1]

        elif op_arr[index, 0] == op.H:

            if clip_h_r > 0:
                raise RuntimeError('Duplicate H records (right) at operation {}'.format(index))

            clip_h_r = op_arr[index, 1]

        elif op_arr[index, 0] == op.M:

            if not allow_m:
                raise RuntimeError('CIGAR op "M" is not allowed')

            if clip_s_r > 0 or clip_h_r > 0:
                raise RuntimeError(
                    'Found clipped bases before last non-clipped CIGAR operation at operation {} ({}{})'.format(
                        index, op_arr[index, 1], op_arr[index, 0]
                    )
                )

            ref_bp += op_arr[index, 1]
            qry_bp += op_arr[index, 1]

        else:
            raise RuntimeError(f'Bad CIGAR op: {op_arr[index, 0]}')

        index += 1

    return ref_bp, qry_bp, clip_h_l, clip_s_l, clip_h_r, clip_s_r


def clip_op_tuples_soft_to_hard(
        op_tuples: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """
    Get a list of CIGAR tuples with soft-clipping converted to hard-clipping.

    :param op_tuples: List of alignment operations as tuples (op_code: int, op_len: int).

    :return: Operation tuples with soft-clipping converted to hard-clipping.
    """

    front_n = 0

    while len(op_tuples) > 0 and op_tuples[0][0] in op.CLIP_SET:
        front_n += op_tuples[0][1]
        op_tuples = op_tuples[1:]

    back_n = 0

    while len(op_tuples) > 0 and op_tuples[-1][0] in op.CLIP_SET:
        back_n += op_tuples[-1][1]
        op_tuples = op_tuples[:-1]

    if len(op_tuples) == 0:
        if front_n + back_n == 0:
            raise RuntimeError('Cannot convert soft clipping to hard: No CIGAR records')

        op_tuples = [(op.H, front_n + back_n)]

    else:
        if front_n > 0:
            op_tuples = [(op.H, front_n)] + op_tuples

        if back_n > 0:
            op_tuples = op_tuples + [(op.H, back_n)]

    return op_tuples
