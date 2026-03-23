"""Call variants per haplotype.

Identifies each variant type using multiple sources for calls:

    * intra: Intra-alignment variants are called from alignment operations.
    * inter: Inter-alignment variants are called from patterns of multiple alignment records broken across large SVs.
"""

import collections
import os
from pathlib import Path
import tarfile
import tempfile
import traceback

import agglovar
import polars as pl
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
            vartype=pav3.vcf.VARTYPES,
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
        csi='{asm_name}.vcf.gz.csi',
    benchmark: 'log/benchmark/{asm_name}/call_vcf.tsv'
    threads: POLARS_MAX_THREADS
    run:
        pattern_var = 'results/{asm_name}/call/call_{vartype}.parquet'
        pattern_callable = 'results/{asm_name}/call_hap/callable_ref_{hap}.parquet'

        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        ref_fa = pav3.pipeline.expand_fofn(input.ref_fofn)[0]

        df_ref_info = pl.read_parquet(input.ref_info)

        # Get haplotypes
        hap_list = pav3.vcf.resolve_hap_list(
            pav3.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE),
            pav_params.vcf_haplotypes,
        )

        if not hap_list:
            raise ValueError(
                f'No haplotypes defined for assembly "{wildcards.asm_name}"'
            )

        hap_order = [
            (wildcards.asm_name, hap) for hap in hap_list
        ]

        # Add callable table for each genotype
        callable_dict = {
            (wildcards.asm_name, hap): pl.scan_parquet(
                pattern_callable.format(asm_name=wildcards.asm_name, hap=hap)
            )
            for hap in hap_list
        }

        # Get genotype tables
        with pav3.io.TempDirContainer(
                prefix=f'pav_vcf_{wildcards.asm_name}_'
        ) as temp_file_container:
            for vartype in pav3.vcf.VARTYPES:

                # Read variants
                df = (
                    pl.scan_parquet(
                        pattern_var.format(
                            asm_name=wildcards.asm_name,
                            vartype=vartype,
                        )
                    )
                    .with_row_index('_index')
                )

                if vartype == 'snv' and df.collect_schema().get('vartype') is None:
                    df = df.with_columns(pl.lit('SNV').alias('vartype'))

                # Initialize VCF fields (all but FORMAT and sample columns)
                try:
                    df = pav3.vcf.init_vcf_fields(
                        df,
                        ref_fa=ref_fa,
                        use_sym=None,
                        vartype=vartype,
                    )
                except Exception as e:
                    raise RuntimeError(f'Failed to initialize VCF fields for {vartype}: {e}') from e

                # Create a sample column
                df = df.with_columns(
                    pl.lit([])
                    .cast(pl.List(pl.String))
                    .alias('_vcf_sample_0')
                )

                # Append genotypes
                try:
                    df = (
                        df.join(
                            pav3.vcf.gt_column_asm(
                                df,
                                wildcards.asm_name,
                                callable_dict,
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
                except Exception as e:
                    raise RuntimeError(f'Failed to append genotypes for {vartype}: {e}') from e

                # Add INFO fields
                try:
                    df = pav3.vcf.standard_info_fields(df)
                except Exception as e:
                    raise RuntimeError(f'Failed to add INFO fields for {vartype}: {e}') from e

                # Finalize VCF fields
                try:
                    df = pav3.vcf.reformat_vcf_table(
                        df,
                        sample_columns={0: wildcards.asm_name}
                    )

                    # Write VCF
                    df.sink_parquet(temp_file_container.next())
                except Exception as e:
                    raise RuntimeError(f'Failed to finalize VCF fields for {vartype}: {e}') from e

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

        pysam.bcftools.index('--csi', output.vcf)


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
    benchmark: 'log/benchmark/{asm_name}/call_callable_{hap}.tsv'
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
            vartype=('insdel', 'inv', 'snv', 'cpx', 'dup')
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
    benchmark: 'log/benchmark/{asm_name}/call_merge_{vartype}.tsv'
    threads: POLARS_MAX_THREADS
    run:
        with open(input.ref_fofn) as in_file:
            ref_path = Path(next(in_file).strip())

        callsets = tuple(
            tuple((
                (
                    pl.scan_parquet(
                        f'results/{wildcards.asm_name}/call_hap/call_{wildcards.vartype}_{hap}.parquet'
                    )
                    .with_row_index('_index')
                ),
                f'{wildcards.asm_name}-{hap}',
            ))
            for hap in pav3.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE)
        )

        pav3.workflow.call.merge_haplotypes(
            vartype=wildcards.vartype,
            callsets=callsets,
            ref_path=ref_path,
            out_path=Path(output.pq),
            merge_params=None,
            merge_name=f'{wildcards.asm_name}_{wildcards.vartype}',
            temp_file_container=None,
        )


# Merge all samples and variant types
localrules: call_tables_hap_all

rule call_tables_hap_all:
    input:
        pq_merge=lambda wildcards: pav3.pipeline.expand_pattern(
            'results/{asm_name}/call_hap/call_{vartype}_{hap}.parquet',
            ASM_TABLE, PAV_CONFIG,
            vartype=('insdel', 'inv', 'snv', 'cpx', 'dup')
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
    benchmark: 'log/benchmark/{asm_name}/call_integrate_sources_{hap}.tsv'
    threads: POLARS_MAX_THREADS
    run:
        pav3.workflow.call.integrate_sources(
            pav_params=pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE),
            input=dict(input),
            output=dict(output),
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
        log='log/call/{asm_name}/call_hap/inter_call_{hap}.log'
    benchmark: 'log/benchmark/{asm_name}/call_inter_{hap}.tsv'
    threads: POLARS_MAX_THREADS
    run:

        # Get parameters
        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        ref_fa_filename, ref_fai_filename = pav3.pipeline.expand_fofn(input.ref_fofn)[:2]
        qry_fa_filename, qry_fai_filename = pav3.pipeline.expand_fofn(input.qry_fofn)[:2]

        score_model = pav3.align.score.get_score_model(pav_params.align_score_model)

        min_anchor_score = pav3.lgsv.chain.get_min_anchor_score(pav_params.min_anchor_score, score_model)

        sort_expr = pav3.call.expr.sort_expr(has_id=False)

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
            temp_dir_parent = f'temp/{wildcards.asm_name}/call_hap/inter'

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
                .sort(sort_expr)
                .sink_parquet(output.pq_insdel)
            )

            # TODO: Fix hom_ref and hom_qry for inversions
            df_inv = (
                pl.DataFrame(
                    df_list_inv, schema=schema_inv
                )
                .lazy()
                .select(schema_inv.keys())
                .sort(sort_expr)
                .sink_parquet(output.pq_inv)
            )

            df_cpx = (
                pl.DataFrame(
                    df_list_cpx, schema=schema_cpx
                )
                .lazy()
                .select(schema_cpx.keys())
                .sort(sort_expr)
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
    benchmark: 'log/benchmark/{asm_name}/call_intra_inv_{hap}.tsv'
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
            ref_fa_filename=ref_fa_filename,
            qry_fa_filename=qry_fa_filename,
            df_ref_fai=df_ref_fai,
            df_qry_fai=df_qry_fai,
            pav_params=pav_params,
        )

        df_inv.write_parquet(output.pq_inv)


# Identify candidate loci for intra-alignment inversions
rule call_intra_inv_flag:
    input:
        pq_snv='temp/{asm_name}/call_hap/intra_snv_{hap}.parquet',
        pq_insdel='temp/{asm_name}/call_hap/intra_insdel_{hap}.parquet',
        ref_fofn='data/ref/ref.fofn',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
    output:
        pq_flag=temp('temp/{asm_name}/call_hap/intra_inv_flagged_sites_{hap}.parquet'),
    benchmark: 'log/benchmark/{asm_name}/call_intra_inv_flag_{hap}.tsv'
    run:
        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        df_ref_fai = agglovar.fa.read_fai(
            pav3.pipeline.expand_fofn(input.ref_fofn)[1]
        ).lazy()

        df_qry_fai = agglovar.fa.read_fai(
            pav3.pipeline.expand_fofn(input.qry_fofn)[1],
            name='qry_id'
        ).lazy()

        # Read
        df_snv = pl.scan_parquet(input.pq_snv)
        df_insdel = pl.scan_parquet(input.pq_insdel)

        # Call per query
        (
            pav3.call.intra.variant_flag_inv(
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
    benchmark: 'log/benchmark/{asm_name}/call_intra_snv_insdel_{hap}.tsv'
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
