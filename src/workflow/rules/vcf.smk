"""
Rules for writing VCF output.
"""

import pavlib

global VCF_PATTERN
global get_config

global shell


_VCF_SVTYPE = {
    'sv': ('ins', 'del', 'inv'),
    'indel': ('ins', 'del'),
    'snv': ('snv', )
}

_VCF_INPUT_PATTERN_BED = 'results/{asm_name}/bed_merged/{filter}/{vartype_svtype}.bed.gz'

_VCF_INPUT_PATTERN_FA = 'results/{asm_name}/bed_merged/{filter}/fa/{vartype_svtype}.fa.gz'

# Make VCF file.
rule vcf_write_vcf:
    input:
        bed_pass=lambda wildcards: pavlib.pipeline.expand_pattern(
            _VCF_INPUT_PATTERN_BED, ASM_TABLE, config,
            merge='merged', filter='pass',
            varsvtype=('snv_snv', 'svindel_ins', 'svindel_del', 'sv_inv', 'sv_cpx')
        ),
        bed_fail=lambda wildcards: pavlib.pipeline.expand_pattern(
            _VCF_INPUT_PATTERN_BED, ASM_TABLE, config,
            merge='merged', filter='pass',
            varsvtype=('snv_snv', 'svindel_ins', 'svindel_del', 'sv_inv')
        ),
        fa=lambda wildcards: pavlib.pipeline.expand_pattern(
            _VCF_INPUT_PATTERN_FA, ASM_TABLE, config,
            merge='merged', filter='fail',
            varsvtype=('svindel_ins', 'svindel_del', 'sv_cpx')
        ),
        fa_fail=lambda wildcards: pavlib.pipeline.expand_pattern(
            _VCF_INPUT_PATTERN_FA, ASM_TABLE, config,
            merge='merged', filter='fail',
            varsvtype=('svindel_ins', 'svindel_del')
        ),
        ref_tsv='data/ref/contig_info.tsv.gz'
    output:
        vcf='{asm_name}.vcf.gz'
    # wildcard_constraints:
    #     merge='merged|hap'
    run:

        merge = 'merged'

        # Get a dictionary of input files.
        #
        # input_dict:
        #   * key: Tuple of...
        #     * [0]: varsvtype
        #     * [1]: "pass" or "fail"
        #   * Value: Tuple of...
        #     * [0]: Variant BED file name.
        #     * [1]: Variant FASTA file name (None if no variant sequences are not used in the VCF).
        input_dict = dict()

        for varsvtype in ('svindel_ins', 'svindel_del', 'sv_inv', 'snv_snv', 'sv_cpx'):

            # Pass
            input_dict[(varsvtype, 'pass')] = (
                _VCF_INPUT_PATTERN_BED.format(
                    asm_name=wildcards.asm_name, merge=merge, filter='pass', varsvtype=varsvtype
                ),
                _VCF_INPUT_PATTERN_FA.format(
                    asm_name=wildcards.asm_name, merge=merge, filter='pass', varsvtype=varsvtype
                ) if varsvtype in {'svindel_ins', 'svindel_del', 'sv_cpx'} else None
            )

            # Fail
            input_dict[(varsvtype, 'fail')] = (
                _VCF_INPUT_PATTERN_BED.format(
                    asm_name=wildcards.asm_name, merge=merge, filter='fail', varsvtype=varsvtype
                ) if varsvtype != 'sv_cpx' else input_dict[(varsvtype, 'pass')][0].iloc[:0, :].copy(),
                _VCF_INPUT_PATTERN_FA.format(
                    asm_name=wildcards.asm_name, merge=merge, filter='fail', varsvtype=varsvtype
                ) if varsvtype in {'svindel_ins', 'svindel_del'} else None
            )

        # Write VCF
        pavlib.vcf.write_merged_vcf(
            asm_name=wildcards.asm_name,
            input_dict=input_dict,
            output_filename=output.vcf,
            ref_tsv=input.ref_tsv,
            reference_filename=get_config(wildcards, 'reference')
        )

        # Write tabix index if possible
        try:
            shell("""tabix {output.vcf} && touch -r {output.vcf} {output.vcf}.tbi""")
        except:
            pass
