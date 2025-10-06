"""CLI entrypoint for all commands.

This module is the entrypoint for all PAV CLI commands. It handles command-line arguments and starts the pipeline.
"""

import argparse
import multiprocessing
import snakemake
from typing import Any, Optional

def parse_arguments(
        argv: Optional[list[Any]] = None
):
    """Parse command-line arguments.

    :param argv: Array of arguments. Defaults to `sys.argv`.

    :return: A configured argument object.

    """

    parser = argparse.ArgumentParser(
        description='PAV: Calls variants from assemblies',
    )

    parser.add_argument(
        '--table', '-t',
        type=str,
        help='Table of input assemblies.'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file.'
    )

    parser.add_argument(
        'targets',
        nargs='*',
        type=str,
        help='PAV targets to run (callset for all samples if none specified).'
    )

    return parser.parse_args(argv)


def main(
        argv: Optional[list[Any]] = None
) -> int:
    """PAV CLI entrypoint.

    :param argv: Array of arguments. Defaults to `sys.argv`.

    :return: Exit code.
    """
    args = parse_arguments(argv)

    print(args)

    snakemake.snakemake(
        cores=multiprocessing.cpu_count()
    )

    return 0
