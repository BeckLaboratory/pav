"""License subcommand."""

import argparse

from .. import __license__
from .. import __license_spdx__
from .. import __license_text__

def subcommand_license(
        out_format: str = 'brief'
) -> int:
    """
    Print license information.

    :param out_format: License format. One of 'brief', 'full', or 'spdx'.

    :returns: Numeric return code (0 for success).
    """
    if out_format == 'brief':
        print(__license__)
    elif out_format == 'full':
        print(__license_text__)
    elif out_format == 'spdx':
        print(__license_spdx__)
    else:
        raise ValueError(f'Unknown license format: {format}')

    return 0

def _add_subparser_license(
        subparsers
) -> argparse.ArgumentParser:
    """Add license subcommand to parser.

    :param subparsers: Subparser object.

    :returns: Configured subparser.
    """
    parser_license = subparsers.add_parser(
        'license',
        help='Show license information.',
    )

    parser_license.add_argument(
        'out_format',
        metavar='out_format',  # Do not shadow "format" builtin in subcommand routine
        type=str,
        choices=['brief', 'full', 'spdx'],
        default='brief',
        help='License output format.',
    )

    return parser_license
