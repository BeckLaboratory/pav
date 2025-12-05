"""
Data files including reference and data tables for the reference.
"""

import os
from pathlib import Path

import agglovar
import pysam

import pav3

global ASM_TABLE
global PAV_CONFIG
global POLARS_MAX_THREADS
global shell


#
# Rules
#

# Generate all pre-target runs
localrules: data_init

rule data_init:
    input:
        ref_fofn='data/ref/ref.fofn',
        ref_info='data/ref/ref_info.parquet'


# Get FASTA files.
rule align_get_qry_fa:
    input:
        fa=lambda wildcards: pav3.pipeline.get_rule_input_list(
            wildcards.asm_name, wildcards.hap, ASM_TABLE
        )
    output:
        fofn='data/query/{asm_name}/query_{hap}.fofn'
    run:

        pav_params = pav3.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        input_tuples = pav3.pipeline.expand_input(
            pav3.pipeline.get_asm_input_list(wildcards.asm_name, wildcards.hap, ASM_TABLE)
        )[0]

        if len(input_tuples) == 0:
            raise ValueError(f'No input sources: {wildcards.asm_name} {wildcards.hap}')

        # Report input sources
        if pav_params.verbose:
            if input_tuples is not None:
                for file_name, file_format in input_tuples:
                    print(f'Input: {wildcards.asm_name} {wildcards.hap}: {file_name} ({file_format})')

        # Link or generate a single FASTA
        out_filename = f'data/query/{wildcards.asm_name}/query_{wildcards.hap}.fa'  # ".gz" is appended as needed

        if len(input_tuples) == 1 and input_tuples[0][1] == 'fasta':
            os.makedirs(os.path.dirname(out_filename), exist_ok=True)
            fa_path_list = pav3.pipeline.link_fasta(input_tuples[0][0], out_filename)

        else:
            # Merge/write FASTA from multiple sources and/or GFA files
            out_filename += '.gz'
            pav3.pipeline.input_tuples_to_fasta(input_tuples, out_filename)

            pysam.faidx(out_filename)

            fa_path_list = [
                Path(out_filename),
                Path(out_filename + '.fai'),
                Path(out_filename + '.gzi')
            ]

        # Write FOFN
        fa_path_list = [str(fa_path) for fa_path in fa_path_list]

        with open(output.fofn, 'w') as f:
            f.write('\n'.join(fa_path_list) + '\n')


# Reference info table
rule data_ref_info_table:
    input:
        fofn='data/ref/ref.fofn'
    output:
        pq='data/ref/ref_info.parquet'
    threads: POLARS_MAX_THREADS
    run:

        ref_fa = pav3.pipeline.expand_fofn(input.fofn)[0]

        agglovar.fa.fa_info(
            ref_fa
        ).write_parquet(
            output.pq
        )


# Prepare reference FASTA.
#
# Creates an FOFN file with two entries, the first is always the path to the reference FASTA file, and the second is
# A path to its index file. If the reference FASTA was missing the index, it is linked to "data/ref/ref.fa" (or with
# ".gz" appended if gzipped) and indexed from there. Otherwise, the FOFN file contains paths to the reference FASTA
# files specified in the PAV config.
rule data_ref_fofn:
    output:
        fofn='data/ref/ref.fofn'
    run:

        # Check reference
        ref_fa = PAV_CONFIG.get('reference', None)

        if ref_fa is None:
            raise ValueError('Missing reference FASTA file in config (')

        ref_fa = str(ref_fa).strip()

        if not os.path.isfile(ref_fa):
            raise FileNotFoundError(f'Reference FASTA file is missing or not a regular file: {ref_fa}')

        if os.stat(ref_fa).st_size == 0:
            raise FileNotFoundError(f'Empty reference FASTA file: {ref_fa}')

        # Link FASTA files
        fa_path_list = pav3.pipeline.link_fasta(ref_fa, 'data/ref/ref.fa')

        # Write FOFN
        fofn_list = [str(fa_path) for fa_path in fa_path_list]

        with open(output.fofn, 'w') as f:
            f.write('\n'.join(fofn_list) + '\n')
