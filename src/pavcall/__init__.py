"""PAV Python library routines."""

__version__ = '3.0.0.dev3'

__all__ = [
    'align',
    'anno',
    'call',
    'cli',
    # 'fig',
    'lgsv',
    'anno',
    # 'asmstat',
    'const',
    'inv',
    'io',
    'kde',
    'params',
    'pipeline',
    'region',
    'schema',
    'seq',
    'util',
    # 'vcf',
]

import importlib

for name in __all__:
    globals()[name] = importlib.import_module(f'.{name}', package=__name__)
