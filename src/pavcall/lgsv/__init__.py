"""Large alignment-truncating variant discovery.

Examines patterns of alignments that truncate around structural variants and determines the best variant call spanning
a set of alignments, if one can be found.
"""

__all__ = [
    'call',
    'chain',
    'interval',
    'io',
    'region_kde',
    'resources',
    'struct',
    'variant',
]

import importlib

for name in __all__:
    globals()[name] = importlib.import_module(f'.{name}', package=__name__)
