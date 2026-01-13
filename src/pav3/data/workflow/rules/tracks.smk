"""PAV-generated tracks"""

import pav3
import shutil
from pathlib import Path

import agglovar
import polars as pl

global ASM_TABLE
global PAV_CONFIG


localrules: tracks_all

rule tracks_all:
    input:
        align=lambda wildcards: pav3.pipeline.expand_pattern(
            'tracks/align/trim_qryref/tracks_align_{asm_name}_trim-qryref.bb',
            ASM_TABLE, PAV_CONFIG,
        ),


#
# Variant tracks
#

rule tracks_var_insdel:
    input:
        pq='results/{asm_name}/call_hap/call_insdel_{hap}.parquet',
        ref_fofn='data/ref/ref.fofn',
    output:
        bb='tracks/call_hap/tracks_call_hap_{asm_name}_{hap}_{varclass}_insdel.bb',
    wildcard_constraints:
        varclass='sv|indel|svindel'
    run:
        ref_fai = pav3.pipeline.expand_fofn(input.ref_fofn)[1]

        df_fai = agglovar.fa.read_fai(ref_fai).lazy()

        bb_tool = shutil.which('bedToBigBed')

        if not bb_tool:
            raise FileNotFoundError('bedToBigBed not found')

        df, as_lines = pav3.fig.tracks.call_hap_insdel(
            {
                'asm_name': wildcards.asm_name,
                'hap': wildcards.hap,
                'callset_path': Path(input.pq),
                'varclass': wildcards.varclass,
                'vartype': 'insdel',
            }
        )

        with pav3.io.TempDirContainer(
                prefix=f'pav_vcf_{wildcards.asm_name}_'
        ) as temp_container:
            temp_bed = temp_container.next(suffix='.bed')
            temp_as = temp_container.next(suffix='.as')

            with open(temp_as, 'wt') as out_file:
                out_file.write('\n'.join(as_lines))

            (
                df.fill_null('.')
                .write_csv(temp_bed, separator='\t', include_header=False)
            )

            shell(
                """{bb_tool} -tab -as={temp_as} -type=bed9+ {temp_bed} {ref_fai} {out_filename}"""
            )



#
# Alignment tracks
#

localrules: tracks_align_all

rule tracks_align_all:
    input:
        bb=lambda wildcards: pav3.pipeline.expand_pattern(
            'tracks/align/trim_qryref/tracks_align_{asm_name}_trim-qryref.bb',
            ASM_TABLE, PAV_CONFIG,
        ),

rule tracks_align:
    input:
        pq=lambda wildcards: pav3.pipeline.expand_pattern(
            'results/{asm_name}/align/{hap}/align_trim-{trim}.parquet',
            ASM_TABLE, PAV_CONFIG,
            asm_name=(wildcards.asm_name),
            trim=('none', 'qry', 'qryref'),
        ),
        ref_fofn='data/ref/ref.fofn',
    output:
        bb_none='tracks/align/trim_none/tracks_align_{asm_name}_trim-none.bb',
        bb_qry='tracks/align/trim_qry/tracks_align_{asm_name}_trim-qry.bb',
        bb_qryref='tracks/align/trim_qryref/tracks_align_{asm_name}_trim-qryref.bb',
    run:
        ALIGN_PATTERN = 'results/{asm_name}/align/{hap}/align_trim-{trim}.parquet'

        ref_fai = pav3.pipeline.expand_fofn(input.ref_fofn)[1]

        bb_tool = shutil.which('bedToBigBed')

        if not bb_tool:
            raise FileNotFoundError('bedToBigBed not found')

        with pav3.io.TempDirContainer(
                prefix=f'pav_vcf_{wildcards.asm_name}_'
        ) as temp_container:
            for trim in ('none', 'qry', 'qryref'):
                out_filename = output[f'bb_{trim}']

                df, as_lines = pav3.fig.tracks.align(
                    [
                        {
                            'asm_name': wildcards.asm_name,
                            'hap': hap,
                            'align_path': Path(ALIGN_PATTERN.format(hap=hap, trim=trim, **wildcards)),
                            'trim': trim,
                        }
                        for hap in pav3.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE)
                    ]
                )

                temp_bed = temp_container.next(suffix='.bed')
                temp_as = temp_container.next(suffix='.as')

                with open(temp_as, 'wt') as out_file:
                    out_file.write('\n'.join(as_lines))

                df.write_csv(temp_bed, separator='\t', include_header=False)

                shell(
                    """{bb_tool} -tab -as={temp_as} -type=bed9+ {temp_bed} {ref_fai} {out_filename}"""
                )
