"""
Data files including reference and data tables for the reference.
"""

import collections

import agglovar
import polars as pl
import pysam

import pavcall

global ASM_TABLE
global PAV_CONFIG
global shell


#
# Pre-run targets
#

def data_init_targets(wildcards=None):
    """
    Get a list of input files to be generated before running samples. This target can be used to setup so that each
    sample can be run independently.

    :param wildcards: Ignored. Function signature needed for Snakemake input function.

    :return: List of run targets.
    """

    part_set = set()

    # Get a set of aligners and partitions
    for asm_name in ASM_TABLE['name']:

        pav_params = pavcall.params.PavParams(asm_name, PAV_CONFIG, ASM_TABLE)

        part_set.add(pav_params.cigar_partitions)
        part_set.add(pav_params.merge_partitions)

    # Construct target list
    target_list = [
        'data/ref/ref.fofn',
        'data/ref/ref_info.parquet'
    ]

    for part_count in part_set:
        target_list.append(
            f'data/ref/partition_{part_count}.tsv.gz'
        )

    return target_list


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
        fa=lambda wildcards: pavcall.pipeline.get_rule_input_list(
            wildcards.asm_name, wildcards.hap, ASM_TABLE
        )
    output:
        fofn='data/query/{asm_name}/query_{hap}.fofn'
    run:

        pav_params = pavcall.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        input_tuples = pavcall.pipeline.expand_input(
            pavcall.pipeline.get_asm_input_list(wildcards.asm_name, wildcards.hap, ASM_TABLE)
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
            fa_path_list = pavcall.pipeline.link_fasta(input_tuples[0][0], out_filename)

        else:
            # Merge/write FASTA from multiple sources and/or GFA files
            out_filename += '.gz'
            pavcall.pipeline.input_tuples_to_fasta(input_tuples, out_filename)

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


# Partition table
#
# Create a table of reference sequences divided into partitions of approximately equal size
localrules: data_ref_partition

rule data_ref_partition:
    input:
        pq='data/ref/ref_info.parquet'
    output:
        pq='data/ref/partition_{part_count}.parquet'
    run:

        part_count = int(wildcards.part_count)

        if part_count < 1:
            raise RuntimeError(f'Number of partitions must be at least 1: {part_count}')

        # Read and sort
        df = (
            pl.read_parquet(input.pq)
            .select(['chrom', 'len'])
            .sort('len', descending=True)
        )

        partition = {chrom: -1 for chrom in df['chrom']}

        # Get a list of assignments for each partition
        list_chr = collections.defaultdict(list)
        list_size = collections.Counter()

        def get_smallest():
            """
            Get the next smallest bin.
            """

            min_index = 0

            for i in range(part_count):

                if list_size[i] == 0:
                    return i

                if list_size[i] < list_size[min_index]:
                    min_index = i

            return min_index

        for chrom in df['chrom']:
            i = get_smallest()
            partition[chrom] = i
            list_size[i] += df.row(by_predicate=pl.col('chrom') == chrom, named=True)['len']

        # Check
        if any([partition[chrom] < 0 for chrom in df['chrom']]):
            raise RuntimeError('Failed to assign all reference contigs to partitions (PROGRAM BUG)')

        # Write parts
        (
            pl.DataFrame({
                'chrom': partition.keys(),
                'partition': partition.values()
            })
            .write_parquet(output.pq)
        )


# Reference info table
rule data_ref_info_table:
    input:
        fofn='data/ref/ref.fofn'
    output:
        pq='data/ref/ref_info.parquet'
    run:

        ref_fa = pavcall.pipeline.expand_fofn(input.fofn)[0]

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
        fa_path_list = pavcall.pipeline.link_fasta(ref_fa, 'data/ref/ref.fa')

        # Write FOFN
        fofn_list = [str(fa_path) for fa_path in fa_path_list]

        with open(output.fofn, 'w') as f:
            f.write('\n'.join(fofn_list) + '\n')
