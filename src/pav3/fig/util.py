"""Figure generation utility functions."""

__all__ = [
    'lighten_color',
    'get_colors_pass_fail',
    'color_to_ucsc_string',
]


import colorsys
from types import ModuleType
from typing import (
    Optional,
    Tuple,
    overload
)

try:
    import matplotlib as mpl
except ImportError as _e:  # pragma: no cover
    mpl: Optional[ModuleType] = None
    _mpl_import_error = _e
else:
    _mpl_import_error = None


def require_matplotlib() -> ModuleType:
    if mpl is None:
        raise ModuleNotFoundError(
            "matplotlib is required for pav3.fig: "
            "Install the optional dependency group, e.g. `pip install 'pav3[fig]'"
        ) from _mpl_import_error

    return mpl


def lighten_color(
        color: str | Tuple[float, float, float],
        amount: float=0.25
) -> Tuple[float, float, float]:
    """Lighten color.

    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Using code from:
    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    mpl_local = require_matplotlib()

    try:
        c = mpl_local.colors.cnames[color]
    except KeyError:
        c = color

    c = colorsys.rgb_to_hls(*mpl_local.colors.to_rgb(c))

    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_colors_pass_fail(
        color_dict: dict[str, tuple[float, float, float]],
        lighten_amt: float=0.25
) -> dict[tuple[str, bool], tuple[float, float, float]]:
    """
    Get a dictionary of colors for PASS/FAIL records. Takes a dictionary of colors with keys describing the color
    (arbitrary string) and values of (R, G, B) tuples (0.0-1.0 values). Returns a dictionary with tuple keys with
    the original name and "True" for PASS and "False" for FAIL. For example, if the input dict has color "COLOR1",
    the output dict has keys ("COLOR1", True) and ("COLOR1", False) where "True" is the original color and "False"
    is an altered version of the color (lightened).

    :param color_dict: Input color dictionary.
    :param lighten_amt: Lightend FAIL coloors by this amount (0.0-1.0).

    :returns: Output color dictionary with tuple keys ("COLOR", True) for PASS records and ("COLOR", False) for FAIL
        records for each "COLOR" in `color_dict`.
    """

    color_dict_out = {
        (key, True): color for key, color in color_dict.items()
    }

    for key, color in color_dict.items():
        color_dict_out[(key, False)] = lighten_color(color, lighten_amt)

    return color_dict_out


@overload
def color_to_ucsc_string(
        color: tuple[float, float, float]
) -> str:
    ...

@overload
def color_to_ucsc_string(
        color: dict[str, tuple[float, float, float]]
) -> dict[str, str]:
    ...

def color_to_ucsc_string(
        color: tuple[float, float, float] | dict[str, tuple[float, float, float]]
) -> str | dict[str, str]:
    """
    Convert a matplotlib color to a UCSC color string.

    :param color: Tuple of RGB values (0.0-1.0) or a dict with tuples of RGB values.

    :returns: Color string (if `color` is a tuple) or color dictionary of strings (if `color` is a dictionary of
        tuple values).
    """

    if isinstance(color, dict):
        return {
            key: ','.join((str(int(color_val * 255)) for color_val in color_val_tup))
                for key, color_val_tup in color.items()
        }

    return ','.join((str(int(color_val * 255)) for color_val in color))
