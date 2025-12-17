"""Variant calling workflow tasks"""

from collections.abc import Iterable
import itertools
import logging
import os
from pathlib import Path
import polars as pl
from typing import (
    Any,
    Optional,
)

import agglovar

from .. import const

from ..io import TempDirContainer

logger = logging.getLogger(__name__)

def merge_haplotypes(
        vartype: str,
        callsets: Iterable[agglovar.merge.base.CallsetDefType],
        ref_path: str | Path,
        out_path: str | Path,
        merge_params: Optional[str | Iterable[dict[str, Any]]] = None,
        merge_name: Optional[str] = None,
        temp_file_container: Optional[TempDirContainer] = None,
):
    """Create a single callset for a phased assembly from all haplotypes.

    :param vartype: Variant type. Used to make some decisions about merging beyond the merge parameters.
    :param callsets: An iterable of callsets to merge. Each element is a two-element tuple with a
        LazyFrame (first element) and the name of the source  callset (second element). The source
        callset is typically "NAME_HAP" (assembly name and haplotype), but can be anything. A default
        source name is set if missing.
    :param ref_path: Path to reference file. Must have a ".fai" index.
    :param out_path: Path to output file.
    :param merge_params: Parameters controlling this merge. If not set, parameters are retrieved from
        `pav3.const.DEFAULT_MERGE_PARAMS` with key `vartype` ("ins" and "del" are aliases for "insdel").
    :param merge_name: Name of this merge for tracking in log messages. If not set, omitted from log messages.
    :param temp_file_container: Manages temporary files written by the merge. If not set, a default container
        is used.
    """
    known_vartype = ['ins', 'del', 'insdel', 'snv', 'inv', 'cpx', 'dup']

    if vartype is None or (vartype := vartype.lower().strip()) not in known_vartype:
        raise ValueError(f'Unknown variant type "{vartype}" (expected one of {", ".join(known_vartype)})')

    if merge_params is None:
        merge_params = const.DEFAULT_MERGE_PARAMS[
            vartype if vartype not in {'ins', 'del'} else 'insdel'
        ]
    else:
        merge_params = list(merge_params)

    ref_path = Path(ref_path)

    if not ref_path.is_file():
        raise FileNotFoundError(f'Reference Path does not exist or is not a regular file: {ref_path}')

    ref_fai_path = Path(str(ref_path) + '.fai')

    if not ref_fai_path.is_file():
        raise FileNotFoundError(f'Reference Path to FAI index does not exist or is not a regular file: {ref_fai_path}')

    callsets = tuple(callsets)

    if merge_name is not None and (merge_name := str(merge_name).strip()):
        log_prefix = f'{merge_name}: '
    else:
        merge_name = None
        log_prefix = ''

    if out_path is None:
        raise ValueError('Output file is missing')

    out_path = Path(out_path)

    os.makedirs(out_path.parent, exist_ok=True)

    # Setup join object
    pairwise_join = agglovar.pairwise.overlap.PairwiseOverlap.from_definiton(
        merge_params
    )

    merge_runner = agglovar.merge.cumulative.MergeCumulative(
        pairwise_join,
        lead_strategy=agglovar.merge.cumulative.LeadStrategy.LEFT,
    )

    # Get chromosome list
    with open(ref_fai_path, 'rt') as in_file:
        chrom_list = sorted([
            chrom for chrom in (
                line.split('\t')[0].strip() for line in in_file
            ) if chrom
        ])

    # Get filters for variant type
    if vartype == 'insdel':
        vartype_list = ['INS', 'DEL']
    else:
        vartype_list = [None,]

    if temp_file_container is None:
        prefix_name = 'pav3_merge_hap'
        if merge_name is not None:
            prefix_name += f'_{merge_name}'

        temp_file_container = TempDirContainer(prefix=prefix_name)

    with temp_file_container:
        concat_chrom_list = []

        for chrom in chrom_list:
            df_chrom_list = []

            for vartype, filter_pass in itertools.product(vartype_list, (True, False)):
                logger.debug('%sMerging chromosome %s (pass=%s): ' % (log_prefix, chrom, filter_pass))

                pre_filter = [
                    pl.col('chrom') == chrom,
                    pl.concat_list(
                        *(['chrom', 'pos', 'end'] if vartype != 'snv' else ['chrom', 'pos', 'alt'])
                    ).is_first_distinct()
                ]

                if filter_pass:
                    pre_filter.append(pl.col('filter').list.len() == 0)
                else:
                    pre_filter.append(pl.col('filter').list.len() > 0)

                if vartype != None:
                    pre_filter.append(pl.col('vartype').str.to_uppercase() == vartype.upper())

                next_filename = temp_file_container.next(
                    prefix=f'split_{chrom}_{vartype if vartype else vartype}_{filter}_'
                )

                df_chrom_list.append(next_filename)

                (
                    merge_runner(
                        callsets,
                        pre_filter=pre_filter,
                    )
                    .with_columns(agglovar.util.var.id_version_expr())
                    .sort('chrom', 'pos', 'end', 'id')
                    .sink_parquet(next_filename)
                )

            # Merge and sort this chromosome
            chrom_next_filename = temp_file_container.next(
                prefix=f'concat_{chrom}_'
            )

            concat_chrom_list.append(chrom_next_filename)

            (
                pl.concat([
                    pl.scan_parquet(filename)
                    for filename in df_chrom_list
                ])
                .sort('chrom', 'pos', 'end', 'id')
                .sink_parquet(chrom_next_filename)
            )

        # Merge all
        logger.debug('%sConcatenating chromosomes' % log_prefix)

        (
            pl.concat([
                pl.scan_parquet(filename)
                for filename in concat_chrom_list
            ])
            .sink_parquet(out_path)
        )

        logger.debug('%sDone' % log_prefix)
