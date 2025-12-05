"""CLI entrypoint for all commands.

This module is the entrypoint for all PAV CLI commands. It handles command-line arguments and starts the pipeline.
"""

import argparse
import inspect
from typing import Any, Optional

from ._subcommand_license import _add_subparser_license, subcommand_license
from ._subcommand_call import _add_subparser_call, subcommand_call
from ._subcommand_batch import _add_subparser_batch, subcommand_batch
from ._common_opt import _add_opt_version

def parse_arguments(
        argv: Optional[list[Any]] = None
):
    """Parse command-line arguments.

    :param argv: Array of arguments. Defaults to `sys.argv`.

    :return: A configured argument object.
    """

    # Create base argument parser
    parser = argparse.ArgumentParser(
        prog='pav3',
        description='PAV assembly-based variant caller',
        epilog=f'The PAV subcommand must be first followed by options and arguments. Use subcommand "call" to call variants.',
    )

    _add_opt_version(parser)

    # Create subparsers
    subparsers = parser.add_subparsers(
        title='subcommands',
        help='PAV subcommand (select one)',
        dest='subcommand',
        required=True,
    )

    # Command: call
    _add_subparser_call(subparsers)
    _add_subparser_batch(subparsers)
    _add_subparser_license(subparsers)

    # Parse
    return parser.parse_args(argv)


def main(
        argv: Optional[list[Any]] = None
) -> int:
    """PAV CLI entrypoint.

    :param argv: Array of arguments. Defaults to `sys.argv`.

    :return: Exit code.
    """
    args = parse_arguments(argv)

    if hasattr(args, 'verbose') and hasattr(args, 'debug'):
        args.verbose = args.verbose or args.debug

    if hasattr(args, 'debug') and args.debug:
        print(f'Running PAV subcommand: {args.subcommand}', flush=True)

    if args.subcommand == 'call':
        return subcommand_call(
            **{
                attr: getattr(args, attr)
                    for attr in list(inspect.signature(subcommand_call).parameters.keys())
            }
        )

    if args.subcommand == 'batch':
        return subcommand_batch(
            **{
                attr: getattr(args, attr)
                    for attr in list(inspect.signature(subcommand_batch).parameters.keys())
            }
        )

    elif args.subcommand == 'license':
        return subcommand_license(
            **{
                attr: getattr(args, attr)
                    for attr in list(inspect.signature(subcommand_license).parameters.keys())
            }
        )
    else:
        raise ValueError(f'Unknown subcommand: {args.subcommand}')
