"""Variant calling workflow tasks"""

from collections import defaultdict
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

import pav3
import agglovar

from .. import const

from ..io import TempDirContainer

from ..call.expr import sort_expr

logger = logging.getLogger(__name__)

def integrate_sources(
        pav_params,
        input: dict[str, str],
        output: dict[str, str],
) -> None:
    inv_min = pav_params.inv_min
    inv_max = pav_params.inv_max if pav_params.inv_max > 0 else float('inf')

    # Read alignments
    df_align_none = pl.scan_parquet(input['align_none'])
    df_align_qry = pl.scan_parquet(input['align_qry'])
    df_align_qryref = pl.scan_parquet(input['align_qryref'])

    # Read trimmed regions (regions are tuples of alignment records and coordinates within the record).
    # IntervalTree where coordinates are tuples - (index, pos):(index, end)
    df_trim = pav3.call.integrate.read_trim_table(
        df_align_none,
        df_align_qry,
        df_align_qryref,
    ).lazy()

    del df_align_none

    # Create a table of regions added to the DISCORD filter
    df_discord_schema = {
        key: val for key, val in pav3.schema.VARIANT.items() if key in {'chrom', 'pos', 'end', 'id'}
    }

    df_discord = pl.LazyFrame([], schema=df_discord_schema)

    # Save alignment records for INNER variants
    df_inner_schema = {
        'align_index': pav3.schema.ALIGN['align_index'],
        'id': pav3.schema.VARIANT['id'],
    }

    df_inner = pl.LazyFrame([], schema=df_inner_schema)

    # Table connecting var_index to ids
    df_var_index_id_schema = {
        'var_index': pav3.schema.VARIANT['var_index'],
        'id': pav3.schema.VARIANT['id'],
    }

    df_var_index_id = pl.LazyFrame([], schema=df_var_index_id_schema)

    # Read segment table
    df_segment = pl.scan_parquet(input['inter_segment'])

    # List of variant tables to collect from multiple sources
    collect_list = defaultdict(list)

    # Parameters controlling how variants are integrated and in what order
    param_dict = {  #    do_write, add_discord, filter_discord, filter_inner
        'inter_cpx':    (True,     True,        False,          False),
        'inter_insdel': (False,    True,        False,          False),
        'inter_inv':    (False,    True,        False,          False),
        'intra_inv':    (True,     True,        True,           True),
        'intra_insdel': (True,     False,       True,           True),
        'intra_snv':    (True,     False,       True,           True),
    }

    # do_write: Write variant call table. Set to False for INS/DEL or INV variants until they are all collected
    # add_discord: Add variant regions to the DISCORD regions.
    # filter_discord: Apply DISCORD filter.
    # filter_inner: Apply INNER filter.
    #
    # Note: add_inner is implied by vartype == 'cpx'

    for sourcetype_vartype in param_dict.keys():
        # if sourcetype_vartype == 'intra_snv':
        #     raise RuntimeError(f'Stopping at {sourcetype_vartype}')

        if pav_params.debug:
            print(f'Processing {sourcetype_vartype}')

        do_write, add_discord, filter_discord, filter_inner = param_dict[sourcetype_vartype]

        sourcetype, vartype = sourcetype_vartype.rsplit('_', 1)

        is_lg = sourcetype == 'inter'
        is_insdel, is_inv, is_snv, is_cpx = (vartype == val for val in ('insdel', 'inv', 'snv', 'cpx'))

        # Read variant table
        df = pav3.call.integrate.set_base_cols(
            pl.scan_parquet(input[sourcetype_vartype]),
            sourcetype,
            vartype,
        )

        # Apply variant length filters
        if is_inv:
            df = (
                df
                .with_columns(
                    pl.when((pl.col('varlen') < inv_min) | (pl.col('varlen') > inv_max))
                    .then(pl.col('filter').list.concat([pl.lit('VARLEN')]))
                    .otherwise(pl.col('filter'))
                    .alias('filter')
                )
            )

        # Filter TRIMREF & TRIMQRY
        if not (is_lg or is_inv):
            df = pav3.call.integrate.apply_trim_filter(df, df_trim)

        # Filter DISCORD
        if filter_discord or filter_inner:
            df = pav3.call.integrate.apply_discord_and_inner_filter(
                df,
                df_discord if filter_discord else None,
                df_inner if filter_inner else None,
            )

        # Version variant IDs prioritizing PASS over non-PASS.
        df = pav3.call.integrate.id_and_version(
            df=df,
            is_snv=is_snv,
            existing_ids=collect_list[vartype]
        )

        # Read CPX segment table
        if sourcetype == 'inter':
            df_segment_var = df_segment.join(df.select(['var_index', 'id']), on='var_index', how='left')
        else:
            df_segment_var = None

        # Get discord expr
        if add_discord and not pav_params.redundant_callset:
            update_discord_frame = (
                pl.concat(
                    [
                        df_discord,
                        (
                            df
                            .filter(
                                pl.col('filter').list.len() == 0,
                                pl.col('end') > (pl.col('pos') + 1)
                            )
                            .select(['chrom', 'pos', 'end', 'id'])
                        )
                    ]
                )
                .sort(['chrom', 'pos', 'end'])
            )
        else:
            update_discord_frame = df_discord

        # Add variants derived from partial complex events
        if is_cpx:
            pav3.call.integrate.add_cpx_derived(
                df=df,
                df_segment=df_segment_var,
                collect_list=collect_list,
            )

        # Aggregate if type is split over multiple inputs
        if not do_write or len(collect_list[vartype]) > 0:
            # Collect here, avoid re-collecting for ID (used as existing IDs) and write.
            # Note: May consume significant memory for some callsets, consider writing to temp and re-reading lazy
            collect_list[vartype].append(df.collect().lazy())

        # Write
        write_list = [update_discord_frame]
        collect_index_discord = len(write_list) - 1

        if sourcetype == 'inter':
            write_list.append(df.select(['var_index', 'id']))
            collect_index_var_index = len(write_list) - 1
        else:
            collect_index_var_index = None

        if df_segment_var is not None:
            write_list.append(
                df_segment_var
                .filter(
                    ~ pl.col('is_anchor')
                    & pl.col('is_aligned')
                    & pl.col('align_index').is_not_null()
                    & pl.col('id').is_not_null()
                )
                .join(  # Passing variants only
                    (
                        df
                        .filter(pl.col('filter').list.len() == 0)
                        .select('var_index')
                    ),
                    on='var_index',
                    how='inner'
                )
                .select(
                    'align_index', 'id'
                )
            )
            collect_index_inner = len(write_list) - 1
        else:
            collect_index_inner = None

        if do_write:
            if collect_list[vartype]:
                df = pl.concat(collect_list[vartype], how='diagonal')

            # Sort and order columns
            col_names = df.collect_schema().names()

            df = (
                df
                .with_row_index('_concat_index')
                .with_columns(
                    pl.col('filter').list.unique().list.sort()
                )
                .sort(
                    pl.col('filter').list.len(),
                    '_concat_index',
                    descending=(True, False),
                )
            )

            df = pav3.call.integrate.id_and_version(
                df=df, is_snv=is_snv,
            )

            df = (
                df
                .sort(pav3.call.expr.sort_expr())
                .select([col for col in pav3.schema.VARIANT.keys() if col in col_names])
            )

            if 'inner' in col_names:
                df = df.with_columns(pl.col('inner').fill_null([]))

            if 'discord' in col_names:
                df = df.with_columns(pl.col('discord').fill_null([]))

            # Create output objects
            write_list.append(df.sink_parquet(output[vartype], lazy=True))

        # Collect and write
        # Always run even if not do_write to update discord regions
        collect_all_list = pl.collect_all(write_list)

        df_discord = collect_all_list[0].lazy()

        if collect_index_var_index is not None:
            df_var_index_id = pl.concat(
                [
                    df_var_index_id.collect(),
                    collect_all_list[collect_index_var_index],
                ]
            ).lazy()

        if collect_index_inner is not None:
            df_inner = pl.concat(
                [
                    df_inner.collect(),
                    collect_all_list[collect_index_inner],
                ]
            ).lazy()

    # Write duplications
    if pav_params.debug:
        print(f'Writing dup')

    df = pl.concat(collect_list['dup'], how='diagonal')

    (
        df
        .with_columns(pl.col('filter').list.unique().sort())
        .sort(pav3.call.expr.sort_expr())
        .select([col for col in pav3.schema.VARIANT.keys() if col in df.collect_schema().names()])
        .sink_parquet(output['dup'])
    )

    # Write segment and ref_trace tables
    if pav_params.debug:
        print(f'Writing segment & trace tables')

    (
        df_segment
        .join(df_var_index_id, on='var_index', how='left')
        .sink_parquet(output['inter_segment'])
    )

    (
        pl.scan_parquet(input['inter_ref_trace'])
        .join(df_var_index_id, on='var_index', how='left')
        .sink_parquet(output['inter_ref_trace'])
    )


def merge_haplotypes(
        vartype: str,
        callsets: Iterable[agglovar.merge.base.CallsetDefType],
        ref_path: str | Path,
        out_path: str | Path,
        merge_params: Optional[str | Iterable[dict[str, Any]]] = None,
        merge_name: Optional[str] = None,
        temp_file_container: Optional[TempDirContainer] = None,
) -> None:
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

    merge_sort_expr = sort_expr(has_id=False)

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
        vartype_list = [vartype.upper(),]

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
                logger.debug('%sMerging chromosome %s (vartype=%s, pass=%s): ' % (log_prefix, chrom, vartype, filter_pass))

                pre_filter = [
                    pl.col('chrom') == chrom,
                    pl.col('vartype').str.to_uppercase() == vartype,
                    pl.concat_list(
                        *(['chrom', 'pos', 'end', 'varlen'] if vartype != 'SNV' else ['chrom', 'pos', 'alt'])
                    ).is_first_distinct(),
                    (
                        pl.col('filter').list.len() == 0
                        if filter_pass else
                        pl.col('filter').list.len() > 0
                    )
                ]

                next_filename = temp_file_container.next(
                    prefix=f'split_{chrom}_{vartype if vartype else vartype}_{filter_pass}_'
                )

                df_chrom_list.append(next_filename)

                (
                    merge_runner(
                        callsets,
                        retain_index=True,
                        pre_filter=pre_filter,
                    )
                    .sort(merge_sort_expr)
                    .with_columns(agglovar.util.var.id_version_expr())
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
