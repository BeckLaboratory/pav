"""Align query sequences and create tables of alignment records.

Query sequences are aligned to the reference and pulled into a table of alignment records with schema
`pav3.schema.ALIGN`. Alignment trimming is applied to resolve redundantly-aligned query sequences (i.e. same bases
of a query sequence in more than one alignment record), then trimmed again to eliminate reference redundancy (i.e.
multiple alignment records covering the same reference bases). Variant discovery will use all three alignment tables
(no trimming, query trimmed, query & reference trimmed) to make calls.
"""


import Bio.bgzf
import gzip
import os

import agglovar
import pav3
import snakemake.io
import polars as pl

global PAV_CONFIG
global expand
global shell
global temp
global get_config
global get_override_config
global ASM_TABLE
global PIPELINE_DIR
global POLARS_MAX_THREADS
global REF_FA
global REF_FAI


#
# Definitions
#

def align_index_files(
        wildcards: snakemake.io.Namedlist | dict[str, str]
):
    """
    Get a list of reference index files needed by an aligner.

    Args:
        wildcards: Rule wildcards.

    Returns:
        Alignment index file list.
    """

    aligner = pav3.params.PavParams(wildcards['asm_name'], PAV_CONFIG, ASM_TABLE).aligner

    # Known aligners
    if aligner == 'minimap2':
        return []

    # New aligners: Fill in indices and create rules to generate the indices.

    # Default to no index files
    return []


#
# Rules
#

# Run all alignments
localrules: align_all

rule align_all:
    input:
        align_trim_none=lambda wildcards: pav3.pipeline.expand_pattern(
            'results/{asm_name}/align/{hap}/align_trim-qryref.parquet', ASM_TABLE, PAV_CONFIG
        )


# Create a depth BED file for alignments.
rule align_depth:
    input:
        pq='results/{asm_name}/align/{hap}/align_trim-{trim}.parquet',
        ref_fofn='data/ref/ref.fofn'
    output:
        pq='results/{asm_name}/align/depth/depth_{side}_{hap}_trim-{trim}_filt-{filt}.parquet'
    wildcard_constraints:
        side=r'qry|ref',
        filt=r'retain|drop'
    threads: POLARS_MAX_THREADS
    run:
        pav3.align.tables.align_depth_table(
            df=pl.read_parquet(input.pq),
            df_fai=agglovar.fa.read_fai(pav3.pipeline.expand_fofn(input.ref_fofn)[1]),
            coord_cols=wildcards.side,
            retain_filtered=wildcards.filt == 'retain'
        ).write_parquet(output.pq)

# Get alignment BED for one part (one aligned cell or split BAM) in one assembly.
rule align_tables:
    input:
        sam='temp/{asm_name}/align/trim-none/align_qry_{hap}.sam.gz',
        ref_fofn='data/ref/ref.fofn',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn'
    output:
        pq_none='results/{asm_name}/align/{hap}/align_trim-none.parquet',
        pq_qry='results/{asm_name}/align/{hap}/align_trim-qry.parquet',
        pq_qryref='results/{asm_name}/align/{hap}/align_trim-qryref.parquet',
        align_head='results/{asm_name}/align/{hap}/align_headers.gz'
    benchmark:
        'data/benchmarks/align/align_tables/align_tables_{asm_name}_{hap}.txt'
    threads: POLARS_MAX_THREADS
    run:

        # Get parameters
        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        qry_fa_filename, qry_fai_filename = pav3.pipeline.expand_fofn(input.qry_fofn)[:2]
        qry_fa_filename = str(qry_fa_filename)
        ref_fa_filename = str(pav3.pipeline.expand_fofn(input.ref_fofn)[0])

        df_qry_fai = agglovar.fa.read_fai(qry_fai_filename, name='qry_id')

        # Get score and LC models
        score_model = pav3.align.score.get_score_model(pav_params.align_score_model)
        lc_model = pav3.align.lcmodel.get_model(pav_params.lc_model)

        # Trim-none: Read alignments as a BED file.
        df_none = pav3.align.tables.sam_to_align_table(
            sam_filename=input.sam,
            df_qry_fai=df_qry_fai,
            score_model=score_model,
            lc_model=lc_model,
            ref_fa_filename=ref_fa_filename
        )

        if pav_params.debug:
            try:
                pav3.align.records.check_matched_bases(df_none, ref_fa_filename, qry_fa_filename)
            except Exception as e:
                raise ValueError(f'Failed to check matched bases before trimming: {e}') from e

        # Apply depth filter
        if pav_params.align_trim_max_depth > 0:
            df_none = pav3.align.tables.align_depth_filter(
                df_none,
                df_depth=None,
                max_depth=pav_params.align_trim_max_depth,
                max_overlap=pav_params.align_trim_max_depth_prop
            )

        df_none.write_parquet(output.pq_none)

        # Trim-qry
        df_qry = pav3.align.trim.trim_alignments_qry(
            df=df_none,
            df_qry_fai=df_qry_fai,
            score_model=score_model,
        )

        try:
            for row in df_qry.iter_rows(named=True):
                pav3.align.records.check_record(row, df_qry_fai)
        except Exception as e:
            raise ValueError(f'Failed to check alignment records (qry trimmed): {wildcards.asm_name}-{wildcards.hap}: {e}') from e

        if pav_params.debug:
            try:
                pav3.align.records.check_matched_bases(df_qry, ref_fa_filename, qry_fa_filename)
            except Exception as e:
                raise ValueError(f'Failed to check matched bases post trimming (qry trimmed): {wildcards.asm_name}-{wildcards.hap}: {e}') from e

        df_qry.write_parquet(output.pq_qry)

        # Trim-ref
        df_qryref = pav3.align.trim.trim_alignments_ref(
            df=df_qry,
            df_qry_fai=df_qry_fai,
            score_model=score_model,
            on_qry=pav_params.redundant_callset
        )

        try:
            for row in df_qryref.iter_rows(named=True):
                pav3.align.records.check_record(row, df_qry_fai)
        except Exception as e:
            raise ValueError(f'Failed to check alignment records (qry & ref trimmed): {wildcards.asm_name}-{wildcards.hap}: {e}') from e

        if pav_params.debug:
            try:
                pav3.align.records.check_matched_bases(df_qryref, ref_fa_filename, qry_fa_filename)
            except Exception as e:
                raise ValueError(f'Failed to check matched bases post trimming (qry & ref trimmed): {wildcards.asm_name}-{wildcards.hap}: {e}') from e

        df_qryref.write_parquet(output.pq_qryref)

        # Write SAM headers
        with gzip.open(input.sam, 'rt') as in_file:
            with gzip.open(output.align_head, 'wt') as out_file:

                line = next(in_file)

                while True:

                    if not line.strip():
                        continue

                    if not line.startswith('@'):
                        break

                    out_file.write(line)

                    try:
                        line = next(in_file)
                    except StopIteration:
                        break



# Map query to reference
rule align_map:
    input:
        ref_fofn='data/ref/ref.fofn',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
        align_index=align_index_files
    output:
        sam=temp('temp/{asm_name}/align/trim-none/align_qry_{hap}.sam.gz')
    benchmark:
        'data/benchmarks/align/align_map/align_map_{asm_name}_{hap}.txt'
    threads: 4
    run:

        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        aligner = pav_params.aligner

        ref_fa = str(pav3.pipeline.expand_fofn(input.ref_fofn)[0])
        qry_fa = str(pav3.pipeline.expand_fofn(input.qry_fofn)[0])

        if os.stat(qry_fa).st_size == 0:
            raise RuntimeError(f'Query file is empty: {qry_fa}')

        # Get alignment command
        if aligner == 'minimap2':
            align_cmd = (
                f"""minimap2 """
                    f"""--secondary=no -a -t {threads} --eqx """
                    f"""{pav_params.align_params} """
                    f"""{ref_fa} {qry_fa}"""
            )

        else:
            raise RuntimeError(f'Unknown alignment program (aligner parameter): {wildcards.asm_name}-{wildcards.hap}: {pav_params.aligner}')

        # Run alignment
        if pav_params.verbose:
            print(f'Aligning {wildcards.asm_name}-{wildcards.hap}: {align_cmd}', flush=True)

        with Bio.bgzf.BgzfWriter(output.sam, 'wt') as out_file:
            for line in shell(align_cmd, iterable=True):
                if not line.startswith('@'):
                    line = line.split('\t')
                    line[9] = '*'
                    line[10] = '*'
                    line = '\t'.join(line)

                out_file.write(line)
                out_file.write('\n')


#
# Export alignments (optional feature)
#

def _align_export_all(wildcards):

    if 'trim' in config:
        trim_set = set(config['trim'].strip().split(','))
    else:
        trim_set = {'qryref'}

    if 'export_fmt' in config:
        ext_set = set(config['export_fmt'].strip().split(','))
    else:
        ext_set = {'cram'}

    ext_set = set([ext if ext != 'sam' else 'sam.gz' for ext in ext_set])

    if 'asm_name' in config:
        asm_set = set(config['asm_name'].strip().split(','))
    else:
        asm_set = None

    if 'hap' in config:
        hap_set = set(config['hap'].strip().split(','))
    else:
        hap_set = None

    return pav3.pipeline.expand_pattern(
        'results/{asm_name}/align/export/pav_align_trim-{trim}_{hap}.{ext}',
        ASM_TABLE, config,
        asm_name=asm_set, hap=hap_set, trim=trim_set, ext=ext_set
    )

# Export alignments
localrules: align_export_all

rule align_export_all:
    input:
        cram=_align_export_all


# # Reconstruct CRAM from alignment BED files after trimming redundantly mapped bases (post-cut).
# rule align_export:
#     input:
#         bed='results/{asm_name}/align/trim-{trim}/align_qry_{hap}.bed.gz',
#         fa='data/query/{asm_name}/query_{hap}.fa.gz',
#         align_head='results/{asm_name}/align/trim-none/align_qry_{hap}.headers.gz',
#         ref_fa='data/ref/ref.fa.gz'
#     output:
#         align='results/{asm_name}/align/export/pav_align_trim-{trim}_{hap}.{ext}'
#     run:
#
#         raise NotImplementedError('Update for PAV3')
#
#         SAM_TAG = fr'@PG\tID:PAV-{wildcards.trim}\tPN:PAV\tVN:{pav3.__version__}\tDS:PAV Alignment trimming {pav3.align.trim.TRIM_DESC[wildcards.trim]}'
#
#         if wildcards.ext == 'cram':
#             out_fmt = 'CRAM'
#             do_bgzip = False
#             do_index = True
#             do_tabix = False
#
#         elif wildcards.ext == 'bam':
#             out_fmt = 'BAM'
#             do_bgzip = False
#             do_index = True
#             do_tabix = False
#
#         elif wildcards.ext == 'sam.gz':
#             out_fmt = 'SAM'
#             do_bgzip = True
#             do_index = False
#             do_tabix = True
#
#         else:
#             raise RuntimeError(f'Unknown output format extension: {wildcards.ext}: (Allowed: "cram", "bam", "sam.gz")')
#
#         # Export
#
#         if not do_bgzip:
#             shell(
#                 """python {PIPELINE_DIR}/scripts/reconstruct_sam.py """
#                     """--bed {input.bed} --fasta {input.fa} --headers {input.align_head} --tag "{SAM_TAG}" | """
#                 """samtools view -T {input.ref_fa} -O {out_fmt} -o {output.align}"""
#             )
#         else:
#             shell(
#                 """python3 {PIPELINE_DIR}/scripts/reconstruct_sam.py """
#                     """--bed {input.bed} --fasta {input.fa} --headers {input.align_head} --tag "{SAM_TAG}" | """
#                 """samtools view -T {input.ref_fa} -O {out_fmt} | """
#                 """bgzip > {output.align}"""
#             )
#
#         # Index
#         if do_index:
#             shell(
#                 """samtools index {output.align}"""
#             )
#
#         if do_tabix:
#             shell(
#                 """tabix {output.align}"""
#             )
