"""Common options.

Common options are defined here. This allows each subcommand to explicitly add common options it uses without relying
on global parent parsers. Some subcommands will use common options that others will not.
"""

import argparse

from .. import __version__

def _add_opt_version(parser: argparse.ArgumentParser) -> None:
    """Add version option to parser."""
    parser.add_argument(
        '--version',
        action='version',
        version=f'PAV {__version__}',
        help='Show PAV version and exit.'
    )

def _add_opt_verbose(
        parser: argparse.ArgumentParser
) -> None:
    """Add verbose option to parser."""
    parser.add_argument(
        '--verbose', '-v',
        default=False, action='store_true',
        help='Generate verbose output',
    )

def _add_opt_debug(
        parser: argparse.ArgumentParser
) -> None:
    """Add debug option to parser."""
    parser.add_argument(
        '--debug',
        default=False, action='store_true',
        help='Generate very verbose debugging output'
    )
