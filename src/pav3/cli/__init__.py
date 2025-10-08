"""PAV command-line interface (CLI), argument parsing, and subcommand routines."""

__all__ = [
    'parse_arguments',
    'main',
    'subcommand_call',
    'subcommand_license',
]

from ._cli import (
    parse_arguments, main,
)

from ._subcommand_call import subcommand_call

from ._subcommand_license import subcommand_license
