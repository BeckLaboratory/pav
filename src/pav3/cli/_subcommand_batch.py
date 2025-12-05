"""Call subcommand."""

import argparse
import importlib.resources
import pathlib
from typing import Optional

import snakemake.cli

from ._common_opt import (
    _add_opt_debug,
    _add_opt_verbose,
    _add_opt_version,
)

def subcommand_batch(
        targets: Optional[list[str]],
        config: Optional[str],
        profile: Optional[str],
        verbose: bool = False,
        debug: bool = False,
        cores: Optional[str | int] = None,
        dry_run: bool = False,
        force: bool = False,
        keep_going: bool = False,
        no_temp: bool = False,
) -> int:

    if debug:
        print('PAV command: batch')
        print(f'\t* targets: {targets}')
        print(f'\t* config: {config}')
        print(f'\t* profile: {profile}')
        print(f'\t* verbose: {verbose}')
        print(f'\t* debug: {debug}')
        print(f'\t* dry_run: {dry_run}')
        print(f'\t* force: {force}')
        print(f'\t* keep_going: {keep_going}', flush=True)

    if targets is None:
        targets = []

    targets = [target_str for target in targets if (target_str := target.strip())]

    if profile is None:
        try:
            profile = str(importlib.resources.files('pav3.data.workflow.profiles').joinpath('pav_default'))

            if not pathlib.Path(profile).is_dir():
                profile = None

        except ModuleNotFoundError:
            pass
    else:
        if not pathlib.Path(profile).is_dir():
            raise FileNotFoundError(f'Profile directory {profile} does not exist.')

    config_params = []

    snake_args = [
        '-s', str(importlib.resources.files('pav3.data.workflow').joinpath('Snakefile')),
        '--rerun-incomplete',
        '--rerun-triggers', 'mtime',
    ]

    if cores:
        snake_args.extend([f'--cores', str(cores)])

    if profile:
        snake_args.extend(['--profile', profile])

    if dry_run:
        snake_args.append('--dry-run')

    if keep_going:
        snake_args.append('--keep-going')

    if no_temp:
        snake_args.append('--no-temp')

    if force:
        snake_args.extend(['--force'])

    if config:
        config_params.append(f'config_file={config}')

    if verbose:
        config_params.extend(['verbose=True'])

    if debug:
        config_params.extend(['debug=True'])

    # Run
    snake_args = (
        snake_args
        + targets
        + (
            ['--config'] + config_params
                if config_params else []
        )
    )

    if debug:
        print(f'Snakemake arguments: {snake_args}', flush=True)

    snakemake.cli.main(snake_args)

    return 0


def _add_subparser_batch(
        subparsers,
) -> argparse.ArgumentParser:
    """Add batch subcommand to parser.

    :param subparsers: Subparser object.

    :returns: Configured subparser.
    """

    parser_batch = subparsers.add_parser(
        'batch',
        description=f'Call variants on all assemblies in an assembly table.',
        help='Call variants from assembly table.'
    )

    _add_opt_version(parser_batch)
    _add_opt_debug(parser_batch)
    _add_opt_verbose(parser_batch)

    parser_batch.add_argument(
        '--config',
        type=str,
        help='Specify a JSON configuration file used by all samples. Must include "reference".',
    )

    parser_batch.add_argument(
        '--profile', '-p',
        type=str,
        help='Path to a Snakemake profile for setting advanced Snakemake options.',
    )

    parser_batch.add_argument(
        '--dry-run', '--dryrun', '-n',
        action='store_true',
        help='Do not execute, show what would be done.',
    )

    parser_batch.add_argument(
        '--keep-going', '-k',
        action='store_true',
        help='Continue even if some jobs fail.',
    )

    parser_batch.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force execution of targets.',
    )

    parser_batch.add_argument(
        '--no-temp', '--notemp', '--nt',
        action='store_true',
        help='Ignore temporary file declarations (temp files are not automatically removed).',
    )

    parser_batch.add_argument(
        '--cores', '-c',
        type=str,
        default=None,
        help='Use at most this many cores. If not set or "all", try to use all available cores.',
    )

    parser_batch.add_argument(
        'targets',
        type=str,
        nargs='*',
        help='Run PAV targets. Can include specific file names to generate or aggregation rule names. Runs all samples by default.',
    )


    return parser_batch
