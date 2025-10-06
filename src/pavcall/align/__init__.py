"""Alignment handling routines."""

import importlib

__all__ = [
    'features',
    'lcmodel',
    'lift',
    'op',
    'records',
    'score',
    'tables',
    'trim',
]

for name in __all__:
    globals()[name] = importlib.import_module(f'.{name}', package=__name__)
