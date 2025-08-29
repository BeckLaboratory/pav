"""
Development tools.

This module contains tools for testing PAV development and debugging. These tools are not used by PAV and are not
loaded by default (i.e. "import pav" will not import "pav.dev"). Most of these tools are designed for interactive use,
such as in ipython or Jupyter notebooks.
"""

from . import sm
from . import tables
