"""General utility functions."""

__all__ = [
    'as_bool',
    'collapse_to_set'
]

from typing import Any, Callable, Iterable, Optional

import polars as pl


def as_bool(
        val: Any,
        fail_to_none: bool = False
) -> Optional[bool]:
    """Get a boolean value.

    True values: "true", "1", "yes", "t", "y", True, 1
    False values: "false", "0", "no", "f", "n", False, 0

    String values are case-insensitive.

    :param val: Value to interpret.
    :param fail_to_none: If `True`, return `None` if `val` is not a recognized boolean value (see above).

    :returns: Boolean value representing `val`.

    :raises ValueError: If `val` is not a recognized boolean value and `fail_to_none` is `False`.
    """
    if issubclass(val.__class__, bool):
        return val

    val = str(val).lower()

    if val in {'true', '1', 'yes', 't', 'y'}:
        return True

    if val in {'false', '0', 'no', 'f', 'n'}:
        return False

    if fail_to_none:
        return None

    raise ValueError('Cannot interpret as boolean value: {}'.format(val))

def collapse_to_set(
        to_flatten: Iterable[Any],
        to_type: Optional[Callable] = None
) -> set[Any]:
    """Flatten an iterable and collapse into a set.

    For each element in the iterable, if it is not a tuple or list, `to_type` is applied (if defined) and the element
    is added to the set. Tuple or list elements are recursively unpacked.

    :param to_flatten: Iterable to flatten.
    :param to_type: A function to convert each element to a specific type (e.g. "int" or "float").

    :returns: Set of unique elements.

    :raises ValueError: If a value fails validation through `to_type`.
    """
    to_flatten = list(to_flatten)  # Copy so the original list is not modified
    s = set()

    if to_type is None:
        to_type = _ident

    while len(to_flatten) > 0:
        v = to_flatten.pop()

        if isinstance(v, (tuple, list)):
            to_flatten.extend(v)
        else:
            s.add(to_type(v))

    return s


def _ident(x):
    """Parameter identity function."""
    return x
