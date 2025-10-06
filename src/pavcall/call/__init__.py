"""Variant calling routines for both intra- and inters-alignment variants."""

__all__ = [
    'expr',
    'integrate',
    'intra',
    'util',
]

from . import expr
from . import integrate
from . import intra
from . import util

# import importlib
#
# for name in __all__:
#     globals()[name] = importlib.import_module(f'.{name}', package=__name__)
