"""
Process alignments and alignment tiling paths.
"""


import Bio.bgzf
import gzip
import os
import pandas as pd

import pavlib
import svpoplib

global config
global expand
global shell
global temp
global get_config
global get_override_config
global ASM_TABLE
global PIPELINE_DIR
global REF_FA
global REF_FAI


#
# Rules
#

# Run all alignments
localrules: align_all

rule align_all:
    input:
        bed_align=lambda wildcards: pavlib.pipeline.expand_pattern(
            'results/{asm_name}/align/trim-{trim}/align_qry_{hap}.bed.gz', ASM_TABLE, config,
            trim=('none', 'qry', 'qryref')
        ),
        bed_depth=lambda wildcards: pavlib.pipeline.expand_pattern(
            'results/{asm_name}/align/trim-{trim}/depth_ref_{hap}.bed.gz', ASM_TABLE, config,
            trim=('none', 'qry', 'qryref')
        )

localrules: align_notrim_all
rule align_notrim_all:
    input:
        bed=lambda wildcards: pavlib.pipeline.expand_pattern(
            'results/{asm_name}/align/trim-none/align_qry_{hap}.bed.gz', ASM_TABLE, config
        )

# Create a BED file of low-confidence alignment regions
# rule align_lowconf_bed:
#     input:
#         bed_qry='results/{asm_name}/align/trim-qry/align_qry_{hap}.bed.gz',
#         bed_none='results/{asm_name}/align/trim-none/align_qry_{hap}.bed.gz'
#     output:
#         bed_qry='results/{asm_name}/align/lowconf/align_qry_{hap}_lowconf_qry-coord.bed.gz',
#         bed_ref='results/{asm_name}/align/lowconf/align_qry_{hap}_lowconf_ref-coord.bed.gz'
#     run:
#
#         # Read
#         df_qry = pd.read_csv(input.bed_qry, sep='\t', dtype={'#CHROM': str, 'QRY_ID': str})
#         df_none = pd.read_csv(input.bed_none, sep='\t', dtype={'#CHROM': str, 'QRY_ID': str})
#


# Create a depth BED file for alignments.f
rule align_depth_bed:
    input:
        bed='results/{asm_name}/align/trim-{trim}/align_qry_{hap}.bed.gz'
    output:
        bed='results/{asm_name}/align/trim-{trim}/depth_ref_{hap}.bed.gz'
    run:

        pavlib.align.tables.align_bed_to_depth_bed(
            pd.read_csv(input.bed, sep='\t', dtype={'#CHROM': str, 'QRY_ID': str}),
            svpoplib.ref.get_df_fai(REF_FAI)
        ).to_csv(
            output.bed, sep='\t', index=False, compression='gzip'
        )

# Cut alignment overlaps in reference coordinates
rule align_trim_qryref:
    input:
        bed='results/{asm_name}/align/trim-qry/align_qry_{hap}.bed.gz',
        qry_fai='data/query/{asm_name}/query_{hap}.fa.gz.fai'
    output:
        bed='results/{asm_name}/align/trim-qryref/align_qry_{hap}.bed.gz'
    run:

        pav_params = pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE)

        # Trim alignments
        df = pavlib.align.trim.trim_alignments(
            pd.read_csv(input.bed, sep='\t', dtype={'#CHROM': str, 'QRY_ID': str}),  # Untrimmed alignment BED
            input.qry_fai,  # Path to query FASTA FAI
            match_qry=pav_params.redundant_callset,  # Redundant callset, trim reference space only for records with matching IDs
            mode='ref',
            score_model=pav_params.align_score
        )

        # Write
        df.to_csv(output.bed, sep='\t', index=False, compression='gzip')

# Cut alignment overlaps in query coordinates
rule align_trim_qry:
    input:
        bed='results/{asm_name}/align/trim-none/align_qry_{hap}.bed.gz',
        qry_fai='data/query/{asm_name}/query_{hap}.fa.gz.fai'
    output:
        bed='results/{asm_name}/align/trim-qry/align_qry_{hap}.bed.gz'
    run:

        pav_params = pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE)

        score_model = pavlib.align.score.get_score_model(pav_params.align_score_model)
        df_qry_fai = svpoplib.ref.get_df_fai(input.qry_fai)

        # Trim alignments
        df = pavlib.align.trim.trim_alignments(
            pd.read_csv(input.bed, sep='\t', dtype={'#CHROM': str, 'QRY_ID': str}),  # Untrimmed alignment BED
            df_qry_fai,  # Path to query FASTA FAI
            mode='qry',
            score_model=score_model
        )

        if pav_params.align_agg_min_score < 0.0:
            df_qry_fai = svpoplib.ref.get_df_fai(input.qry_fai)

            df = pavlib.align.tables.aggregate_alignment_records(
                df, df_qry_fai,
                score_model=score_model,
                min_score=pav_params.align_agg_min_score,
                noncolinear_penalty=pav_params.align_agg_noncolinear_penalty,
                assign_order=True
            )

        # Write
        df.to_csv(output.bed, sep='\t', index=False, compression='gzip')

# Get alignment BED for one part (one aligned cell or split BAM) in one assembly.
#
# Note: lcmodel training uses the alignment score model saved in the stats TSV file.
rule align_get_bed:
    input:
        sam='temp/{asm_name}/align/trim-none/align_qry_{hap}.sam.gz',
        qry_fai='data/query/{asm_name}/query_{hap}.fa.gz.fai'
    output:
        bed='results/{asm_name}/align/trim-none/align_qry_{hap}.bed.gz',
        align_head='results/{asm_name}/align/trim-none/align_qry_{hap}.headers.gz',
        tsv_stats='results/{asm_name}/align/trim-none/stats_qry_{hap}.tsv.gz'
    run:

        pav_params = pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE)

        # Read FAI
        df_qry_fai = svpoplib.ref.get_df_fai(input.qry_fai)
        df_qry_fai.index = df_qry_fai.index.astype(str)

        # Get score model
        score_model = pavlib.align.score.get_score_model(pav_params.align_score_model)

        # Get LC model
        lc_model = pavlib.align.lcmodel.get_model(
            pav_params.lc_model,
            os.path.join(PIPELINE_DIR, pavlib.const.PAV_LC_MODEL_SUBDIR)
        )

        # Read alignments as a BED file.
        df = pavlib.align.tables.get_align_bed(
            input.sam, df_qry_fai, wildcards.hap,
            score_model=score_model,
            lc_model=lc_model
        )

        # Add trimming fields
        df['TRIM_REF_L'] = 0
        df['TRIM_REF_R'] = 0
        df['TRIM_QRY_L'] = 0
        df['TRIM_QRY_R'] = 0

        # Write SAM headers
        if os.stat(input.sam).st_size > 0:
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

        # Write
        df.to_csv(output.bed, sep='\t', index=False, compression='gzip')

        # Create stats
        df_pass = df[df['FILTER'] == 'PASS']
        df_fail = df[df['FILTER'] != 'PASS']

        prop_n_pass = (df_pass.shape[0] / df.shape[0]) if df.shape[0] > 0 else 0.0
        prop_n_fail = (df_fail.shape[0] / df.shape[0]) if df.shape[0] > 0 else 0.0

        bp_pass = (df_pass['QRY_END'] - df_pass['QRY_POS']).sum()
        bp_fail = (df_fail['QRY_END'] - df_fail['QRY_POS']).sum()
        bp_all = (df['END'] - df['POS']).sum()

        prop_bp_pass = bp_pass / bp_all if bp_all > 0 else 0.0
        prop_bp_fail = bp_fail / bp_all if bp_all > 0 else 0.0

        df_stats = pd.Series(
            [
                df.shape[0],
                df_pass.shape[0], prop_n_pass, bp_pass, prop_bp_pass,
                df_fail.shape[0], prop_n_fail, bp_fail, prop_bp_fail,
                pav_params.align_score_model
            ],
            index=[
                'N',
                'PASS_N', 'PASS_PROP', 'PASS_BP', 'PASS_BP_PROP',
                'FAIL_N', 'FAIL_PROP', 'FAIL_BP', 'FAIL_BP_PROP',
                'SCORE_MODEL'
            ]
        )

        df_stats.to_csv(output.tsv_stats, sep='\t', index=True, header=False, compression='gzip')


# Map query as SAM. Pull read information from the SAM before sorting and writing CRAM since tool tend to change
# "=X" to "M" in the CIGAR.
rule align_map:
    input:
        ref_fa='data/ref/ref.fa.gz',
        seq_files=lambda wildcards: pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE).get_aligner_input()
    output:
        sam=temp('temp/{asm_name}/align/trim-none/align_qry_{hap}.sam.gz')
    threads: 4
    run:

        pav_params = pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE)

        aligner = pav_params.aligner

        # Get alignment command
        if aligner == 'minimap2':
            align_cmd = (
                f"""minimap2 """
                    f"""{pav_params.align_params} """
                    f"""--secondary=no -a -t {threads} --eqx -Y """
                    f"""{input.ref_fa} {input.seq_files[0]}"""
            )

        elif aligner == 'lra':
            align_cmd = (
                f"""lra align {input.ref_fa} {input.seq_files[0]} -CONTIG -p s -t {threads}"""
                f"""{pav_params.aligner_params}"""
            )

        else:
            raise RuntimeError(f'Unknown alignment program (aligner parameter): {pav_params.aligner}')

        # Run alignment
        if os.stat(input.seq_files[0]).st_size > 0:

            # Run alignment
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

        else:
            # Write an empty file if input is empty
            with open(output.sam, 'w') as out_file:
                pass

# Uncompress query sequences for aligners that cannot read gzipped FASTAs.
rule align_uncompress_qry:
    input:
        fa='data/query/{asm_name}/query_{hap}.fa.gz'
    output:
        fa='temp/{asm_name}/align/query/query_{hap}.fa'
    run:

        if os.stat(input.fa).st_size > 0:
            shell(
                """zcat {input.fa} > {output.fa}"""
            )
        else:
            with open(output.fa, 'w') as out_file:
                pass

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

    return pavlib.pipeline.expand_pattern(
        'results/{asm_name}/align/export/pav_align_trim-{trim}_{hap}.{ext}',
        ASM_TABLE, config,
        asm_name=asm_set, hap=hap_set, trim=trim_set, ext=ext_set
    )

# Get CRAM files
localrules: align_export_all

rule align_export_all:
    input:
        cram=_align_export_all


# Reconstruct CRAM from alignment BED files after trimming redundantly mapped bases (post-cut).
rule align_export:
    input:
        bed='results/{asm_name}/align/trim-{trim}/align_qry_{hap}.bed.gz',
        fa='data/query/{asm_name}/query_{hap}.fa.gz',
        align_head='results/{asm_name}/align/trim-none/align_qry_{hap}.headers.gz',
        ref_fa='data/ref/ref.fa.gz'
    output:
        align='results/{asm_name}/align/export/pav_align_trim-{trim}_{hap}.{ext}'
    run:

        SAM_TAG = fr'@PG\tID:PAV-{wildcards.trim}\tPN:PAV\tVN:{pavlib.const.get_version_string()}\tDS:PAV Alignment trimming {pavlib.align.trim.TRIM_DESC[wildcards.trim]}'

        if wildcards.ext == 'cram':
            out_fmt = 'CRAM'
            do_bgzip = False
            do_index = True
            do_tabix = False

        elif wildcards.ext == 'bam':
            out_fmt = 'BAM'
            do_bgzip = False
            do_index = True
            do_tabix = False

        elif wildcards.ext == 'sam.gz':
            out_fmt = 'SAM'
            do_bgzip = True
            do_index = False
            do_tabix = True

        else:
            raise RuntimeError(f'Unknown output format extension: {wildcards.ext}: (Allowed: "cram", "bam", "sam.gz")')

        # Export

        if not do_bgzip:
            shell(
                """python3 {PIPELINE_DIR}/scripts/reconstruct_sam.py """
                    """--bed {input.bed} --fasta {input.fa} --headers {input.align_head} --tag "{SAM_TAG}" | """
                """samtools view -T {input.ref_fa} -O {out_fmt} -o {output.align}"""
            )
        else:
            shell(
                """python3 {PIPELINE_DIR}/scripts/reconstruct_sam.py """
                    """--bed {input.bed} --fasta {input.fa} --headers {input.align_head} --tag "{SAM_TAG}" | """
                """samtools view -T {input.ref_fa} -O {out_fmt} | """
                """bgzip > {output.align}"""
            )

        # Index
        if do_index:
            shell(
                """samtools index {output.align}"""
            )

        if do_tabix:
            shell(
                """tabix {output.align}"""
            )
