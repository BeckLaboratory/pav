"""Utilities for working with tables."""

__all__ = [
    'show_row',
]

from typing import Any


def show_row(row: dict[str, Any], max_len: int = 80) -> dict[str, Any]:
    """Get a readable table row.

    :param row: Row as a dict.
    :param max_len: Maximum length of string columns.

    :returns: Row transformed for display in a terminal (commas added to positions, long columns eliminated).
    """
    def fmt_val(key, val):

        # Commas
        if key in {'pos', 'end', 'qry_pos', 'qry_end'}:
            return f'{val:,}'

        # Truncate
        if key == 'seq':
            return f"{val[:5]}{'+' if len(val) > 5 else ''}({len(val):,d}bp)" if val is not None else 'Null'

        if key == 'align_ops':
            return f"OPS(count={len(val.get('op_code', []))})"

        if isinstance(val, str) and len(val) > max_len:
            return f"{val[:12]}..."

        # Default (no transformation)
        return val

    return {key: fmt_val(key, val) for key, val in row.items()}
