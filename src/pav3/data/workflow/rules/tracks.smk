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
        call_insdel=lambda wildcards: pav3.pipeline.expand_pattern(
            'tracks/call_hap/tracks_call_hap_{asm_name}_{hap}_{varclass}_insdel.bb',
            ASM_TABLE, PAV_CONFIG,
            varclass=('sv', 'indel')
        ),
        call_snv=lambda wildcards: pav3.pipeline.expand_pattern(
            'tracks/call_hap/tracks_call_hap_{asm_name}_{hap}_snv_snv.bb',
            ASM_TABLE, PAV_CONFIG,
            varclass=('sv', 'indel')
        ),
        call_invdupcpx=lambda wildcards: pav3.pipeline.expand_pattern(
            'tracks/call_hap/tracks_call_hap_{asm_name}_{hap}_sv_invdupcpx.bb',
            ASM_TABLE, PAV_CONFIG,
        ),


#
# Variant tracks
#

rule tracks_var_invdupcpx:
    input:
        pq_inv='results/{asm_name}/call_hap/call_inv_{hap}.parquet',
        pq_dup='results/{asm_name}/call_hap/call_dup_{hap}.parquet',
        pq_cpx='results/{asm_name}/call_hap/call_cpx_{hap}.parquet',
        pq_segment='results/{asm_name}/call_hap/inter/inter_segment_{hap}.parquet',
        ref_fofn='data/ref/ref.fofn',
    output:
        bb='tracks/call_hap/tracks_call_hap_{asm_name}_{hap}_sv_invdupcpx.bb',
    run:

        ref_fai = pav3.pipeline.expand_fofn(input.ref_fofn)[1]
        df_fai = agglovar.fa.read_fai(ref_fai)

        df, as_lines = pav3.fig.tracks.call_hap_invdupcpx(
            {
                'asm_name': wildcards.asm_name,
                'hap': wildcards.hap,
                'callset_path_inv': Path(input.pq_inv),
                'callset_path_dup': Path(input.pq_dup),
                'callset_path_cpx': Path(input.pq_cpx),
                'callset_path_segment': Path(input.pq_segment),
                'varclass': 'sv',
                'vartype': 'invdupcpx',
            },
            df_fai=df_fai,
        )

        with pav3.io.TempDirContainer(
                prefix=f'pav_track_invdupcpx_{wildcards.asm_name}_'
        ) as temp_container:
            temp_bed = temp_container.next(suffix='.bed')
            temp_as = temp_container.next(suffix='.as')

            with open(temp_as, 'wt') as out_file:
                out_file.write('\n'.join(as_lines))

            (
                df
                .cast(pl.String)
                .fill_null('.')
                .sink_csv(temp_bed, separator='\t', include_header=False)
            )

            shell(
                """bedToBigBed -tab -as={temp_as} -type=bed9+ {temp_bed} {ref_fai} {output.bb}"""
            )


rule tracks_var_insdel_snv:
    input:
        pq='results/{asm_name}/call_hap/call_{vartype}_{hap}.parquet',
        ref_fofn='data/ref/ref.fofn',
    output:
        bb='tracks/call_hap/tracks_call_hap_{asm_name}_{hap}_{varclass}_{vartype}.bb',
    wildcard_constraints:
        varclass='sv|indel|svindel|snv',
        vartype='insdel|snv',
    run:
        ref_fai = pav3.pipeline.expand_fofn(input.ref_fofn)[1]
        df_fai = agglovar.fa.read_fai(ref_fai)

        if (wildcards.varclass == 'snv') ^ (wildcards.vartype == 'snv'):
            raise ValueError(f'Variant class ({wildcards.varclass}) and variant type ({wildcards.vartype}) most both be "snv" for SNV variants')

        # bb_tool = shutil.which('bedToBigBed')
        #
        # if not bb_tool:
        #     raise FileNotFoundError('bedToBigBed not found')

        df, as_lines = pav3.fig.tracks.call_hap_insdelsnv(
            {
                'asm_name': wildcards.asm_name,
                'hap': wildcards.hap,
                'callset_path': Path(input.pq),
                'varclass': wildcards.varclass,
                'vartype': wildcards.vartype,
            },
            df_fai=df_fai,
        )

        with pav3.io.TempDirContainer(
                prefix=f'pav_track_insdel_snv_{wildcards.asm_name}_'
        ) as temp_container:
            temp_bed = temp_container.next(suffix='.bed')
            temp_as = temp_container.next(suffix='.as')

            with open(temp_as, 'wt') as out_file:
                out_file.write('\n'.join(as_lines))

            (
                df
                .cast(pl.String)
                .fill_null('.')
                .sink_csv(temp_bed, separator='\t', include_header=False)
            )

            shell(
                """bedToBigBed -tab -as={temp_as} -type=bed9+ {temp_bed} {ref_fai} {output.bb}"""
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

        # bb_tool = shutil.which('bedToBigBed')
        #
        # if not bb_tool:
        #     raise FileNotFoundError('bedToBigBed not found')

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

                df.sink_csv(temp_bed, separator='\t', include_header=False)

                shell(
                    """bedToBigBed -tab -as={temp_as} -type=bed9+ {temp_bed} {ref_fai} {out_filename}"""
                )
