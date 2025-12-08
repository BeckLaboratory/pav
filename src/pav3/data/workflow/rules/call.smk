"""Call variants per haplotype.

Identifies each variant type using multiple sources for calls:

    * intra: Intra-alignment variants are called from alignment operations.
    * inter: Inter-alignment variants are called from patterns of multiple alignment records broken across large SVs.
"""

import collections
import itertools
import os
import tarfile
import tempfile
import traceback

import agglovar
import polars as pl
import pysam
import pysam.bcftools

import pav3

global ASM_TABLE
global PAV_CONFIG
global POLARS_MAX_THREADS
global temp


#
# Rules
#

# Generate the VCF
rule call_vcf:
    input:
        pq_merge=lambda wildcards: pav3.pipeline.expand_pattern(
            'results/{asm_name}/call/call_{vartype}.parquet',
            ASM_TABLE, PAV_CONFIG,
            asm_name=wildcards.asm_name,
            vartype=('insdel', 'inv', 'snv', 'cpx')
        ),
        callable_ref=lambda wildcards: pav3.pipeline.expand_pattern(
            'results/{asm_name}/call_hap/callable_ref_{hap}.parquet',
            ASM_TABLE, PAV_CONFIG,
            asm_name=wildcards.asm_name,
        ),
        ref_fofn='data/ref/ref.fofn',
        ref_info='data/ref/ref_info.parquet',
    output:
        vcf='{asm_name}.vcf.gz',
        tbi='{asm_name}.vcf.gz.csi',
    threads: POLARS_MAX_THREADS
    run:

        pattern_var = 'results/{asm_name}/call/call_{vartype}.parquet'
        pattern_callable = 'results/{asm_name}/call_hap/callable_ref_{hap}.parquet'

        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        ref_fa = pav3.pipeline.expand_fofn(input.ref_fofn)[0]

        df_ref_info = pl.read_parquet(input.ref_info)

        # Get haplotypes
        hap_source = pav3.vcf.get_hap_source(
            pav3.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE),
            pav_params.vcf_haplotypes,
        )

        if not hap_source:
            raise ValueError(
                f'No haplotypes defined for assembly "{wildcards.asm_name}"'
            )

        # Add callable table for each genotype
        hap_callable = {
            hap: pl.scan_parquet(
                pattern_callable.format(asm_name=wildcards.asm_name, hap=hap)
            )
            for hap in hap_source.keys()
        }

        # Get genotype tables
        with pav3.io.TempDirContainer(
                prefix=f'pav_vcf_{wildcards.asm_name}_'
        ) as temp_file_container:
            for vartype in ('insdel', 'inv', 'snv', 'cpx'):

                # Read variants
                df = (
                    pl.scan_parquet(
                        pattern_var.format(
                            asm_name=wildcards.asm_name,
                            vartype=vartype
                        )
                    )
                    .with_row_index('_index')
                )

                if vartype == 'snv':
                    df = df.with_columns(pl.lit('SNV').alias('vartype'))

                # Initialize VCF fields (all but FORMAT and sample columns)
                df = pav3.vcf.init_vcf_fields(
                    df,
                    ref_fa=ref_fa,
                    use_sym=None,
                )

                # Create a sample column
                df = df.with_columns(
                    pl.lit([])
                    .cast(pl.List(pl.String))
                    .alias('_vcf_sample_0')
                )

                # Append genotypes
                df = (
                    df.join(
                        pav3.vcf.gt_column(
                            df,
                            hap_source,
                            hap_callable,
                            col_name='_vcf_field_gt',
                            separator='|',
                        ),
                        on='_index',
                        how='left'
                    )
                    .with_columns(
                        pl.col('_vcf_sample_0').list.concat('_vcf_field_gt'),
                        pl.col('_vcf_format').list.concat(pl.lit('GT'))
                    )
                    .drop('_vcf_field_gt')
                )

                # Add INFO fields
                df = pav3.vcf.standard_info_fields(df)

                # Finalize VCF fields
                df = pav3.vcf.reformat_vcf_table(
                    df,
                    sample_columns={0: wildcards.asm_name}
                )

                # Write VCF
                df.sink_parquet(temp_file_container.next())

            # Write
            header_list = pav3.vcf.get_headers(
                ref_filename=PAV_CONFIG['reference'],
                df_ref_info=df_ref_info,
            )

            df_records = (
                pl.concat(
                    [
                        pl.scan_parquet(str(file_path)) for file_path in temp_file_container.values()
                    ]
                )
                .sort('#CHROM', 'POS', 'ID')
            )

            # with open('deleme.vcf', 'wt') as out_file:
            # with pav3.io.BGZFWriterIO(output.vcf) as out_file:
            # with Bio.bgzf.BgzfWriter(output.vcf, 'wb') as out_file:
            with pav3.io.BGZFWriterIO(output.vcf, encoding='utf-8') as out_file:

                # Headers
                out_file.write('\n'.join([
                    line for line in header_list
                ]))
                out_file.write('\n')

                # Records
                df_records.sink_csv(
                    out_file,
                    separator='\t',
                )

        pysam.bcftools.index(output.vcf)


rule call_tables_callable:
    input:
        align_qryref='results/{asm_name}/align/{hap}/align_trim-qryref.parquet',
        inter_insdel='temp/{asm_name}/call_hap/inter_insdel_{hap}.parquet',
        inter_inv='temp/{asm_name}/call_hap/inter_inv_{hap}.parquet',
        inter_cpx='temp/{asm_name}/call_hap/inter_cpx_{hap}.parquet',
        ref_fofn='data/ref/ref.fofn',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
    output:
        callable_ref='results/{asm_name}/call_hap/callable_ref_{hap}.parquet',
        callable_qry='results/{asm_name}/call_hap/callable_qry_{hap}.parquet',
    run:

        # Read tables
        df_ref_fai = agglovar.fa.read_fai(
            pav3.pipeline.expand_fofn(input.ref_fofn)[1],
        )

        df_qry_fai = agglovar.fa.read_fai(
            pav3.pipeline.expand_fofn(input.qry_fofn)[1],
            name='qry_id'
        )

        df_in = pl.concat([
            (
                pl.scan_parquet(filename)
                .filter(pl.col('filter').list.len() == 0)
                .select(
                    'chrom', 'pos', 'end',
                    'qry_id', 'qry_pos', 'qry_end',
                )
            ) for filename in (
                input.align_qryref,
                input.inter_insdel,
                input.inter_inv,
                input.inter_cpx,

            )
        ])

        # Get callable regions
        (
            pav3.align.tables.align_depth_table(df_in, df_ref_fai, coord_cols='ref')
            .filter(pl.col('depth') > 0)
            .drop('index', 'depth')
            .write_parquet(output.callable_ref)
        )

        (
            pav3.align.tables.align_depth_table(df_in, df_qry_fai, coord_cols='qry')
            .filter(pl.col('depth') > 0)
            .drop('index', 'depth')
            .write_parquet(output.callable_qry)
        )


# Merge all samples and variant types
localrules: call_tables_all

rule call_tables_all:
    input:
        pq_merge=lambda wildcards: pav3.pipeline.expand_pattern(
            'results/{asm_name}/call/call_{vartype}.parquet',
            ASM_TABLE, PAV_CONFIG,
            vartype=('insdel', 'inv', 'snv', 'cpx')
        )

# Merge one sample and variant type
rule call_tables:
    input:
        pq_inter=lambda wildcards: pav3.pipeline.expand_pattern(
            'results/{asm_name}/call_hap/call_{vartype}_{hap}.parquet',
            ASM_TABLE, PAV_CONFIG,
            asm_name=wildcards.asm_name,
            vartype=wildcards.vartype,
        ),
        ref_fofn='data/ref/ref.fofn'
    output:
        pq='results/{asm_name}/call/call_{vartype}.parquet'
    threads: POLARS_MAX_THREADS
    run:

        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        # Get merge params
        try:
            merge_params = pav3.const.DEFAULT_MERGE_PARAMS[wildcards.vartype]
        except KeyError:
            raise ValueError(f'No merge parameters for variant type: {wildcards.vartype}')

        pairwise_join = agglovar.pairwise.overlap.PairwiseOverlap.from_definiton(
            merge_params
        )

        merge_runner = agglovar.merge.cumulative.MergeCumulative(
            pairwise_join,
            lead_strategy=agglovar.merge.cumulative.LeadStrategy.LEFT,
        )

        # Get chromosome list
        with open(input.ref_fofn) as in_file:
            ref_filename = next(in_file).strip()

        with open(ref_filename + '.fai') as in_file:
            chrom_list = sorted([
                chrom for chrom in (
                    line.split('\t')[0].strip() for line in in_file
                ) if chrom
            ])

        # Get filters for variant type
        if wildcards.vartype == 'insdel':
            vartype_list = ['INS', 'DEL']
        else:
            vartype_list = [None,]

        with pav3.io.TempDirContainer(
                prefix=f'pav_call_tables_{wildcards.asm_name}_{wildcards.vartype}_'
        ) as temp_file_container:

            concat_chrom_list = []

            for chrom in chrom_list:
                if pav_params.verbose:
                    print(f'call_tables (asm_name={wildcards.asm_name}, vartype={wildcards.vartype}): Merging chrom "{chrom}"')

                df_chrom_list = []

                for vartype, filter_pass in itertools.product(vartype_list, (True, False)):
                    # print(f'{vartype} - {filter_pass}')

                    filters = [pl.col('chrom') == chrom]

                    if filter_pass:
                        filters.append(pl.col('filter').list.len() == 0)
                    else:
                        filters.append(pl.col('filter').list.len() > 0)

                    if vartype != None:
                        filters.append(pl.col('vartype') == vartype)

                    next_filename = temp_file_container.next(
                        prefix=f'split_{chrom}_{vartype if vartype else wildcards.vartype}_{filter}_'
                    )

                    df_chrom_list.append(next_filename)

                    (
                        merge_runner(
                            (
                                (
                                    pl.scan_parquet(
                                        f'results/{wildcards.asm_name}/call_hap/call_{wildcards.vartype}_{hap}.parquet'
                                    )
                                    .with_row_index('_index')
                                    .filter(*filters)
                                    .with_columns(
                                        pl.col('id').str.replace(r'\..*$', '').alias('_id_base')
                                    )
                                    .unique('_id_base', keep='first')
                                    .drop('_id_base')
                                    ,
                                    f'{wildcards.asm_name}-{hap}',
                                ) for hap in pav3.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE)
                            ),
                            retain_index=True
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
            if pav_params.verbose:
                print(f'call_tables (asm_name={wildcards.asm_name}, vartype={wildcards.vartype}): Concat chroms')

            (
                pl.concat([
                    pl.scan_parquet(filename)
                    for filename in concat_chrom_list
                ])
                .sink_parquet(output.pq)
            )

# Integrate variant sources
rule call_integrate_sources:
    input:
        inter_insdel='temp/{asm_name}/call_hap/inter_insdel_{hap}.parquet',
        inter_inv='temp/{asm_name}/call_hap/inter_inv_{hap}.parquet',
        inter_cpx='temp/{asm_name}/call_hap/inter_cpx_{hap}.parquet',
        intra_inv='temp/{asm_name}/call_hap/intra_inv_{hap}.parquet',
        intra_snv='temp/{asm_name}/call_hap/intra_snv_{hap}.parquet',
        intra_insdel='temp/{asm_name}/call_hap/intra_insdel_{hap}.parquet',
        align_none='results/{asm_name}/align/{hap}/align_trim-none.parquet',
        align_qry='results/{asm_name}/align/{hap}/align_trim-qry.parquet',
        align_qryref='results/{asm_name}/align/{hap}/align_trim-qryref.parquet',
        inter_segment='temp/{asm_name}/call_hap/inter_segment_{hap}.parquet',
        inter_ref_trace='temp/{asm_name}/call_hap/inter_reftrace_cpx_{hap}.parquet',
    output:
        insdel='results/{asm_name}/call_hap/call_insdel_{hap}.parquet',
        inv='results/{asm_name}/call_hap/call_inv_{hap}.parquet',
        cpx='results/{asm_name}/call_hap/call_cpx_{hap}.parquet',
        snv='results/{asm_name}/call_hap/call_snv_{hap}.parquet',
        dup='results/{asm_name}/call_hap/call_dup_{hap}.parquet',
        inter_segment='results/{asm_name}/call_hap/inter/inter_segment_{hap}.parquet',
        inter_ref_trace='results/{asm_name}/call_hap/inter/inter_reftrace_cpx_{hap}.parquet',
    threads: POLARS_MAX_THREADS
    run:

        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        inv_min = pav_params.inv_min
        inv_max = pav_params.inv_max if pav_params.inv_max > 0 else float('inf')

        # Read alignments
        df_align_none = pl.scan_parquet(input.align_none)
        df_align_qry = pl.scan_parquet(input.align_qry)
        df_align_qryref = pl.scan_parquet(input.align_qryref)

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
        df_segment = pl.scan_parquet(input.inter_segment)

        # List of variant tables to collect from multiple sources
        collect_list = collections.defaultdict(list)

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
            df = pl.scan_parquet(input[sourcetype_vartype])

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
                    .select(['align_index', 'id'])
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
                    .sort(['chrom', 'pos', 'end', 'id'])
                    .with_columns(pl.col('filter').list.unique().list.sort())
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
            .sort(['chrom', 'pos', 'end', 'id'])
            .select([col for col in pav3.schema.VARIANT.keys() if col in df.collect_schema().names()])
            .sink_parquet(output.dup)
        )

        # Write segment and ref_trace tables
        if pav_params.debug:
            print(f'Writing segment & trace tables')

        (
            df_segment
            .join(df_var_index_id, on='var_index', how='left')
            .sink_parquet(output.inter_segment)
        )

        (
            pl.scan_parquet(input.inter_ref_trace)
            .join(df_var_index_id, on='var_index', how='left')
            .sink_parquet(output.inter_ref_trace)
        )


# Call alignment-truncating SVs.
rule call_inter:
    input:
        align_none='results/{asm_name}/align/{hap}/align_trim-none.parquet',
        align_qry='results/{asm_name}/align/{hap}/align_trim-qry.parquet',
        align_qryref='results/{asm_name}/align/{hap}/align_trim-qryref.parquet',
        ref_fofn='data/ref/ref.fofn',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
    output:
        pq_insdel=temp('temp/{asm_name}/call_hap/inter_insdel_{hap}.parquet'),
        pq_inv=temp('temp/{asm_name}/call_hap/inter_inv_{hap}.parquet'),
        pq_cpx=temp('temp/{asm_name}/call_hap/inter_cpx_{hap}.parquet'),
        pq_segment=temp('temp/{asm_name}/call_hap/inter_segment_{hap}.parquet'),
        pq_ref_trace=temp('temp/{asm_name}/call_hap/inter_reftrace_cpx_{hap}.parquet'),
        dot_tar='results/{asm_name}/call_hap/inter/inter_graph_{asm_name}_{hap}.tar',
    log:
        log='log/{asm_name}/call_hap/inter_call_{hap}.log'
    threads: POLARS_MAX_THREADS
    run:

        # Get parameters
        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        ref_fa_filename, ref_fai_filename = pav3.pipeline.expand_fofn(input.ref_fofn)[:2]
        qry_fa_filename, qry_fai_filename = pav3.pipeline.expand_fofn(input.qry_fofn)[:2]

        score_model = pav3.align.score.get_score_model(pav_params.align_score_model)

        min_anchor_score = pav3.lgsv.chain.get_min_anchor_score(pav_params.min_anchor_score, score_model)

        # Read alignments
        df_align_qry = pl.scan_parquet(input.align_qry)
        df_align_qryref = pl.scan_parquet(input.align_qryref)
        df_align_none = pl.scan_parquet(input.align_none)

        # Get KDE for inversions
        kde_model = pav3.kde.KdeTruncNorm(
            pav_params.inv_kde_bandwidth, pav_params.inv_kde_trunc_z, pav_params.inv_kde_func
        )

        with open(log.log, 'w') as log_file:

            # Set caller resources
            caller_resources = pav3.lgsv.resources.CallerResources(
                df_align_qry=df_align_qry,
                df_align_qryref=df_align_qryref,
                df_align_none=df_align_none,
                ref_fa_filename=str(ref_fa_filename),
                qry_fa_filename=str(qry_fa_filename),
                ref_fai_filename=str(ref_fai_filename),
                qry_fai_filename=str(qry_fai_filename),
                score_model=score_model,
                k_util=agglovar.kmer.util.KmerUtil(pav_params.inv_k_size),
                kde_model=kde_model,
                log_file=log_file,
                verbose=True,
                pav_params=pav_params,
            )

            # Call
            temp_dir_parent = f'temp/{wildcards.asm_name}/call_hap/intra'

            os.makedirs(temp_dir_parent, exist_ok=True)

            with tempfile.TemporaryDirectory(
                    dir=temp_dir_parent, prefix=f'call_inter_{wildcards.hap}_dotfiles.'
            ) as dot_dirname:

                lgsv_list = pav3.lgsv.call.call_from_align(
                    caller_resources, min_anchor_score=min_anchor_score, dot_dirname=dot_dirname
                )

                with tarfile.open(output.dot_tar, 'w') as tar_file:
                    for file in os.listdir(dot_dirname):
                        tar_file.add(os.path.join(dot_dirname, file))

            # Expected schemas with additional keys (var_index) for cross-table relations
            schema_insdel = {
                col: pav3.schema.VARIANT[col] for col in
                    pav3.schema.VARIANT.keys() if col in (
                        pav3.lgsv.variant.InsertionVariant.row_set({'var_index'}) |
                        pav3.lgsv.variant.DeletionVariant.row_set({'var_index'})
                )
            }

            schema_inv = {
                col: pav3.schema.VARIANT[col] for col in
                    pav3.schema.VARIANT.keys() if col in (
                        pav3.lgsv.variant.InversionVariant.row_set({'var_index'})
                )
            }

            schema_cpx = {
                col: pav3.schema.VARIANT[col] for col in
                    pav3.schema.VARIANT.keys() if col in (
                        pav3.lgsv.variant.ComplexVariant.row_set({'var_index'})
                )
            }

            schema_ref_trace = (
                    pav3.lgsv.struct.REF_TRACE_SCHEMA |
                    {'var_index': pav3.schema.VARIANT['var_index']}
            )

            schema_segment = (
                    pav3.lgsv.interval.SEGMENT_TABLE_SCHEMA |
                    {'var_index': pav3.schema.VARIANT['var_index']}
            )

            # Create tables
            df_list_insdel = []
            df_list_inv = []
            df_list_cpx = []

            df_segment_list = []
            df_reftrace_list = []

            df_list = {
                'INS': df_list_insdel,
                'DEL': df_list_insdel,
                'INV': df_list_inv,
                'CPX': df_list_cpx
            }

            var_index = 0

            # Sort by chrom and qry_id before resolving (calling row()), faster to retrieve sequences and homology
            lgsv_list.sort(key=lambda var: (var.interval.region_ref.chrom, var.interval.region_ref.pos))

            for var in lgsv_list:

                if var.is_null or var.is_patch:
                    continue

                if caller_resources.verbose:
                    print(f'Completing variant: {var}', file=caller_resources.log_file, flush=True)

                var.var_index = var_index
                var_index += 1

                try:
                    row = var.row({'var_index'})
                except Exception as e:
                    traceback.print_exc()

                    raise ValueError(f'Failed to get variant row for "{var}": {e}') from e

                if row['vartype'] not in df_list.keys():
                    raise ValueError(f'Unexpected variant type: "{row["vartype"]}" in "{var}"')

                df_list[row['vartype']].append(row)

                if var.df_segment is not None:
                    df_segment_list.append(
                        var.df_segment
                        .with_columns(pl.lit(row['var_index']).alias('var_index'))
                    )

                if var.df_ref_trace is not None:
                    df_reftrace_list.append(
                        var.df_ref_trace
                        .with_columns(pl.lit(row['var_index']).alias('var_index'))
                    )

            # Collect and write
            (
                pl.DataFrame(
                    df_list_insdel, schema=schema_insdel
                )
                .lazy()
                .select(schema_insdel.keys())
                .sort(['chrom', 'pos', 'end', 'qry_id', 'qry_pos', 'qry_end'])
                .sink_parquet(output.pq_insdel)
            )

            # TODO: Fix hom_ref and hom_qry for inversions
            df_inv = (
                pl.DataFrame(
                    df_list_inv, schema=schema_inv
                )
                .lazy()
                .select(schema_inv.keys())
                .sort(['chrom', 'pos', 'end', 'qry_id', 'qry_pos', 'qry_end'])
                .sink_parquet(output.pq_inv)
            )

            df_cpx = (
                pl.DataFrame(
                    df_list_cpx, schema=schema_cpx
                )
                .lazy()
                .select(schema_cpx.keys())
                .sort(['chrom', 'pos', 'end', 'qry_id', 'qry_pos', 'qry_end'])
                .sink_parquet(output.pq_cpx)
            )

            (
                (pl.concat(df_segment_list) if df_segment_list else pl.DataFrame(schema=schema_segment))
                .lazy()
                .cast(schema_segment)
                .select(schema_segment.keys())
                .sort(['var_index', 'seg_index'])
                .sink_parquet(output.pq_segment)
            )

            (
                (pl.concat(df_reftrace_list) if df_reftrace_list else pl.DataFrame(schema=schema_ref_trace))
                .lazy()
                .cast(schema_ref_trace)
                .select(schema_ref_trace.keys())
                .sort('var_index', maintain_order=True)
                .sink_parquet(output.pq_ref_trace)
            )


rule call_intra_inv:
    input:
        align_none='results/{asm_name}/align/{hap}/align_trim-none.parquet',
        pq_flag='temp/{asm_name}/call_hap/intra_inv_flagged_sites_{hap}.parquet',
        ref_fofn='data/ref/ref.fofn',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
    output:
        pq_inv=temp('temp/{asm_name}/call_hap/intra_inv_{hap}.parquet'),
    threads: POLARS_MAX_THREADS
    run:

        # Get parameters
        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        df_align = pl.scan_parquet(input.align_none)
        ref_fa_filename, ref_fai_filename = pav3.pipeline.expand_fofn(input.ref_fofn)[0:2]
        qry_fa_filename, qry_fai_filename = pav3.pipeline.expand_fofn(input.qry_fofn)[0:2]

        # Read
        df_ref_fai = agglovar.fa.read_fai(ref_fai_filename)
        df_qry_fai = agglovar.fa.read_fai(qry_fai_filename, name='qry_id')
        df_flag = pl.scan_parquet(input.pq_flag)

        # Call per query
        df_inv = pav3.call.intra.variant_tables_inv(
            df_align=df_align,
            df_flag=df_flag,
            ref_fa_filename=str(ref_fa_filename),
            qry_fa_filename=str(qry_fa_filename),
            df_ref_fai=df_ref_fai,
            df_qry_fai=df_qry_fai,
            pav_params=pav_params,
        )

        df_inv.write_parquet(output.pq_inv)


# Identify candidate loci for intra-alignment inversions
rule call_intra_inv_flag:
    input:
        align_none='results/{asm_name}/align/{hap}/align_trim-none.parquet',
        pq_snv='temp/{asm_name}/call_hap/intra_snv_{hap}.parquet',
        pq_insdel='temp/{asm_name}/call_hap/intra_insdel_{hap}.parquet',
        ref_fofn='data/ref/ref.fofn',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
    output:
        pq_flag=temp('temp/{asm_name}/call_hap/intra_inv_flagged_sites_{hap}.parquet'),
    run:

        # Get parameters
        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        df_align = pl.scan_parquet(input.align_none)
        ref_fai_filename = pav3.pipeline.expand_fofn(input.ref_fofn)[1]
        qry_fai_filename = pav3.pipeline.expand_fofn(input.qry_fofn)[1]

        # Read FAI files
        df_ref_fai = agglovar.fa.read_fai(ref_fai_filename).lazy()
        df_qry_fai = agglovar.fa.read_fai(qry_fai_filename, name='qry_id').lazy()

        # Read
        df_snv = pl.scan_parquet(input.pq_snv)
        df_insdel = pl.scan_parquet(input.pq_insdel)

        # Call per query
        (
            pav3.call.intra.variant_flag_inv(
                df_align=df_align,
                df_snv=df_snv,
                df_insdel=df_insdel,
                df_ref_fai=df_ref_fai,
                df_qry_fai=df_qry_fai,
                pav_params=pav_params,
            )
            .write_parquet(output.pq_flag)
        )


# Call intra-alignment SNV and INS/DEL variants
rule call_intra_snv_insdel:
    input:
        align_none='results/{asm_name}/align/{hap}/align_trim-none.parquet',
        ref_fofn='data/ref/ref.fofn',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
    output:
        pq_snv=temp('temp/{asm_name}/call_hap/intra_snv_{hap}.parquet'),
        pq_insdel=temp('temp/{asm_name}/call_hap/intra_insdel_{hap}.parquet')
    threads: POLARS_MAX_THREADS
    run:

        # Get parameters
        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        ref_fa_filename, ref_fai_filename = pav3.pipeline.expand_fofn(input.ref_fofn)[0:2]
        qry_fa_filename, qry_fai_filename = pav3.pipeline.expand_fofn(input.qry_fofn)[0:2]

        # Read
        df_align = pl.scan_parquet(input.align_none)

        # Call and write
        temp_dir_parent = f'temp/{wildcards.asm_name}/call_hap/intra'

        os.makedirs(temp_dir_parent, exist_ok=True)

        with tempfile.TemporaryDirectory(
                dir=temp_dir_parent, prefix=f'call_intra_{wildcards.hap}_snv_insdel.'
        ) as temp_dir_name:

            df_snv, df_insdel = pav3.call.intra.variant_tables_snv_insdel(
                df_align=df_align,
                ref_fa_filename=str(ref_fa_filename),
                qry_fa_filename=str(qry_fa_filename),
                temp_dir_name=temp_dir_name,
                pav_params=pav_params,
            )

            df_snv.sink_parquet(output.pq_snv)
            df_insdel.sink_parquet(output.pq_insdel)
