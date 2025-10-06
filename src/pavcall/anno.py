"""Functions for annotations."""

__all__ = [
    'perfect_homology'
]

import numpy as np
import agglovar.kmer.util

_BASE_TO_INT = np.vectorize(lambda base: np.int8(agglovar.kmer.util.BASE_TO_INT.get(base, -1)), otypes=[np.int32])
"""Private function - convert a base to an integer (-1 for non-ACGT bases)."""


def perfect_homology(
        seq_var: str,
        seq_ref: str,
        pos: int,
        end: int,
        is_rev: bool
) -> tuple[int, int]:
    """Get the number of perfect homology bases between two sequences.

    :param seq_var: Variant sequence in reference orientation.
    :param seq_ref: Reference or query sequence.
    :param pos: Variant position in .
    :param end: Variant end position.
    :param is_rev: Whether the variant is on the reverse strand of seq_ref.

    :returns: A tuple of upstream homology (first element) and downstream homology (second element).
    """
    var = _BASE_TO_INT(list(seq_var))
    var_pass = var != -1

    # Upstream sequence vector
    flank_up = _BASE_TO_INT(list(
        seq_ref[
            max(pos - var.shape[0], 0): pos
        ]
    ))

    flank_up_pass = flank_up != -1

    # Downstream sequence vector
    flank_dn = _BASE_TO_INT(list(
        seq_ref[
            end: end + var.shape[0]
        ]
    ))

    flank_dn_pass = flank_dn != -1

    if is_rev:
        flank_up, flank_dn = ~flank_dn[::-1] & 0x3, ~flank_up[::-1] & 0x3
        flank_up_pass, flank_dn_pass = flank_dn_pass[::-1], flank_up_pass[::-1]

    up_len = flank_up.shape[0]
    dn_len = flank_dn.shape[0]

    return (
        _true_len(
            (var[::-1][:up_len] == flank_up[::-1]) &
            var_pass[::-1][:up_len] &
            flank_up_pass[::-1]
        ),
        _true_len(
            (var[:dn_len] == flank_dn) &
            var_pass[:dn_len] &
            flank_dn_pass
        ),
    )


def _true_len(
        arr: np.ndarray[np.bool_]
) -> int:
    """Get the length of true values from the start of the array to the first False value (or end of the array).

    :param arr: Boolean array.

    :returns: Length of the first run of True values in `arr`.
    """
    return int(false_loc[0] if len(false_loc := np.where(~arr)[0]) > 0 else 0)
