"""
Call variants from aligned contigs.
"""

import agglovar
import os
import tempfile

import polars as pl

import pavcall

global ASM_TABLE
global PAV_CONFIG
global temp


#
# Inter-alignment variants (large variants)
#

# # Call alignment-truncating SVs.
# rule call_inter:
#     input:
#         align_none='results/{asm_name}/align/{hap}/align_trim-none.parquet',
#         align_qry='results/{asm_name}/align/{hap}/align_trim-qry.parquet',
#         align_qryref='results/{asm_name}/align/{hap}/align_trim-qryref.parquet',
#         ref_fofn='data/ref/ref.fofn',
#         qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
#     output:
#         bed_ins='results/{asm_name}/lgsv/svindel_ins_{hap}.bed.gz',
#         bed_del='results/{asm_name}/lgsv/svindel_del_{hap}.bed.gz',
#         bed_inv='results/{asm_name}/lgsv/sv_inv_{hap}.bed.gz',
#         bed_cpx='results/{asm_name}/lgsv/sv_cpx_{hap}.bed.gz',
#         bed_seg='results/{asm_name}/lgsv/segment_{hap}.bed.gz',
#         bed_cpx_ref='results/{asm_name}/lgsv/reftrace_cpx_{hap}.bed.gz',
#         dot_tar='results/{asm_name}/lgsv/lgsv_graph_{asm_name}_{hap}.tar'
#     log:
#         log='log/{asm_name}/lgsv/lgsv_call_{hap}.log'
#     run:
#
#         # Get parameters
#         pav_params = pavcall.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)
#
#         ref_fa_filename, ref_fai_filename = pavcall.pipeline.expand_fofn(input.ref_fofn)[:2]
#         qry_fa_filename, qry_fai_filename = pavcall.pipeline.expand_fofn(input.qry_fofn)[:2]
#
#         score_model = pavcall.align.score.get_score_model(pav_params.align_score_model)
#
#         # # Set graph file output
#         # dot_dirname = f'temp/{wildcards.asm_name}/lgsv/graph_{wildcards.hap}'
#         # os.makedirs(dot_dirname, exist_ok=True)
#
#         # Get score model
#
#         # Get minimum anchor score
#         min_anchor_score = pavcall.lgsv.util.get_min_anchor_score(pav_params.min_anchor_score, score_model)
#
#         # Read alignments - Trim QRY
#         df_align_qry = pd.read_csv(
#             input.bed_qry,
#             sep='\t',
#             dtype={'#CHROM': str, 'QRY_ID': str}
#         )
#
#         df_align_qry.sort_values(['QRY_ID', 'QRY_ORDER'], inplace=True)
#         df_align_qry.reset_index(inplace=True, drop=True)
#
#         # Read alignments - Trim QRY/REF
#         df_align_qryref = pd.read_csv(
#             input.bed_qryref,
#             sep='\t',
#             index_col='INDEX',
#             dtype={'#CHROM': str, 'QRY_ID': str}
#         )
#
#         # Read alignments - Trim NONE
#         df_align_none = pd.read_csv(
#             input.bed_none,
#             sep='\t',
#             index_col='INDEX',
#             dtype={'#CHROM': str, 'QRY_ID': str}
#         )
#
#         # Get KDE for inversions
#         kde_model = pavlib.kde.KdeTruncNorm(
#             pav_params.inv_kde_bandwidth, pav_params.inv_kde_trunc_z, pav_params.inv_kde_func
#         )
#
#         with open(log.log, 'w') as log_file:
#
#             # Set caller resources
#             caller_resources = pavlib.lgsv.CallerResources(
#                 df_align_qry=df_align_qry,
#                 df_align_qryref=df_align_qryref,
#                 df_align_none=df_align_none,
#                 qry_fa_name=input.fa_qry,
#                 ref_fa_name=input.fa_ref,
#                 hap=wildcards.hap,
#                 score_model=score_model,
#                 k_util=kanapy.util.kmer.KmerUtil(pav_params.inv_k_size),
#                 kde_model=kde_model,
#                 log_file=log_file,
#                 verbose=True,
#                 pav_params=pav_params
#             )
#
#             # Call
#             lgsv_list = pavlib.lgsv.call.call_from_align(caller_resources, min_anchor_score=min_anchor_score, dot_dirname=dot_dirname)
#
#             # Remove duplicate IDs (most likely never actually removes variants, but creates table tracking issues if it does)
#             lgsv_list_dedup = list()
#             lgsv_id_set = set()
#
#             for var in lgsv_list:
#                 if var.variant_id not in lgsv_id_set:
#                     lgsv_list_dedup.append(var)
#                     lgsv_id_set.add(var.variant_id)
#
#             lgsv_list = lgsv_list_dedup
#
#             # Create tables
#             df_list = {
#                 'INS': list(),
#                 'DEL': list(),
#                 'INV': list(),
#                 'CPX': list()
#             }
#
#             for var in lgsv_list:
#
#                 if caller_resources.verbose:
#                     print(f'Completing variant: {var}', file=caller_resources.log_file, flush=True)
#
#                 row = var.row()
#
#                 if row['SVTYPE'] not in df_list.keys():
#                     raise RuntimeError(f'Unexpected SVTYPE: "{row["SVTYPE"]}"')
#
#                 df_list[row['SVTYPE']].append(row)
#
#             if len(df_list['INS']) > 0:
#                 df_ins = pd.concat(df_list['INS'], axis=1).T
#             else:
#                 df_ins = pd.DataFrame([], columns=pavlib.lgsv.variant.InsertionVariant(None, None).row().index)
#
#             if len(df_list['DEL']) > 0:
#                 df_del = pd.concat(df_list['DEL'], axis=1).T
#             else:
#                 df_del = pd.DataFrame([], columns=pavlib.lgsv.variant.DeletionVariant(None, None).row().index)
#
#             if len(df_list['INV']) > 0:
#                 df_inv = pd.concat(df_list['INV'], axis=1).T
#             else:
#                 df_inv = pd.DataFrame([], columns=pavlib.lgsv.variant.InversionVariant(None, None).row().index)
#
#             if len(df_list['CPX']) > 0:
#                 df_cpx = pd.concat(df_list['CPX'], axis=1).T
#             else:
#                 df_cpx = pd.DataFrame([], columns=pavlib.lgsv.variant.ComplexVariant(None, None).row().index)
#
#             df_ins.sort_values(['#CHROM', 'POS', 'END', 'ID', 'QRY_ID', 'QRY_POS', 'QRY_END'], inplace=True)
#             df_del.sort_values(['#CHROM', 'POS', 'END', 'ID', 'QRY_ID', 'QRY_POS', 'QRY_END'], inplace=True)
#             df_inv.sort_values(['#CHROM', 'POS', 'END', 'ID', 'QRY_ID', 'QRY_POS', 'QRY_END'], inplace=True)
#             df_cpx.sort_values(['#CHROM', 'POS', 'END', 'ID', 'QRY_ID', 'QRY_POS', 'QRY_END'], inplace=True)
#
#             # Write variant tables
#             df_ins.to_csv(output.bed_ins, sep='\t', index=False, compression='gzip')
#             df_del.to_csv(output.bed_del, sep='\t', index=False, compression='gzip')
#             df_inv.to_csv(output.bed_inv, sep='\t', index=False, compression='gzip')
#             df_cpx.to_csv(output.bed_cpx, sep='\t', index=False, compression='gzip')
#
#             # Write segment and reference trace tables
#             df_segment_list = list()
#             df_reftrace_list = list()
#
#             for var in lgsv_list:
#
#                 df_segment = var.interval.df_segment.copy()
#                 df_segment.insert(3, 'ID', var.variant_id)
#                 df_segment_list.append(df_segment)
#
#                 if var.svtype == 'CPX':
#                     df_reftrace = var.df_ref_trace.copy()
#                     df_reftrace.insert(3, 'ID', var.variant_id)
#                     df_reftrace_list.append(df_reftrace)
#
#             if len(df_segment_list) > 0:
#                 df_segment = pd.concat(df_segment_list, axis=0)
#             else:
#                 df_segment = pd.DataFrame([], columns=(
#                     pavlib.lgsv.interval.SEGMENT_TABLE_FIELDS[:3] + ['ID'] + pavlib.lgsv.interval.SEGMENT_TABLE_FIELDS[3:]
#                 ))
#
#             if len(df_reftrace_list) > 0:
#                 df_reftrace = pd.concat(df_reftrace_list, axis=0)
#             else:
#                 df_reftrace = pd.DataFrame([], columns=(
#                     pavlib.lgsv.variant.REF_TRACE_COLUMNS[:3] + ['ID'] + pavlib.lgsv.variant.REF_TRACE_COLUMNS[3:]
#                 ))
#
#             df_segment.to_csv(output.bed_seg, sep='\t', index=False, compression='gzip')
#             df_reftrace.to_csv(output.bed_cpx_ref, sep='\t', index=False, compression='gzip')
#
#             # Compress graph dot files
#             if dot_dirname is not None:
#                 with tarfile.open(output.dot_tar,'w') as tar_file:
#                     for file in os.listdir(dot_dirname):
#                         tar_file.add(os.path.join(dot_dirname, file))
#
#             shutil.rmtree(dot_dirname)


#
# Intra-alignment variants
#

# Call intra-alignment SNVs
#
# SNV table is filtered by query trimming (not reference trimming), and
rule call_intra:
    input:
        align_none='results/{asm_name}/align/{hap}/align_trim-none.parquet',
        ref_fofn='data/ref/ref.fofn',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
    output:
        pq_snv=temp('temp/{asm_name}/call/intra_snv_{hap}.parquet'),
        pq_insdel=temp('temp/{asm_name}/call/intra_insdel_{hap}.parquet'),
        pq_inv=temp('temp/{asm_name}/call/intra_inv_{hap}.parquet'),
    run:

        # Get parameters
        pav_params = pavcall.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        ref_fa_filename, ref_fai_filename = pavcall.pipeline.expand_fofn(input.ref_fofn)[0:2]
        qry_fa_filename, qry_fai_filename = pavcall.pipeline.expand_fofn(input.qry_fofn)[0:2]

        # Read FAI files
        df_ref_fai = agglovar.fa.read_fai(ref_fai_filename)
        df_qry_fai = agglovar.fa.read_fai(qry_fai_filename, name='qry_id')

        # Read
        df_align = pl.scan_parquet(input.align_none)

        # Call and write
        temp_dir_parent = f'temp/{wildcards.asm_name}/call/intra'

        os.makedirs(temp_dir_parent, exist_ok=True)

        with tempfile.TemporaryDirectory(
                dir=temp_dir_parent, prefix=f'call_intra_{wildcards.hap}_{wildcards.vartype}.'
        ) as temp_dir_name:

            df_snv, df_insdel, df_inv = pavcall.call.intra.variant_tables(
                df_align=df_align,
                ref_fa_filename=ref_fa_filename,
                qry_fa_filename=qry_fa_filename,
                df_ref_fai=df_ref_fai,
                df_qry_fai=df_qry_fai,
                temp_dir_name=temp_dir_name,
                pav_params=pav_params,
            )

            df_snv.sink_parquet(output.pq_snv)
            df_insdel.sink_parquet(output.pq_insdel)
            df_inv.sink_parquet(output.pq_inv)
