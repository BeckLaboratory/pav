"""
Call inversions from aligned query sequences.

Inversion calling has two key steps:
1) Flag: Find signatures of inversions from alignments and/or variant calls.
2) Call: Use flagged loci to call inversions. Calling starts with the flagged locus, then it expands until it
   finds the full inversion including unique reference sequence on each flank.
"""

import gzip

import agglovar
import polars as pl

import pavcall

global ASM_TABLE
global REF_FA
global PAV_CONFIG
global expand
global temp
global get_config


#
# Definitions
#

# # Column names for the inversion call table
# INV_CALL_COLUMNS = [
#     '#CHROM', 'POS', 'END',
#     'ID', 'SVTYPE', 'SVLEN',
#     'HAP',
#     'QRY_ID', 'QRY_POS', 'QRY_END', 'QRY_STRAND',
#     'CI',
#     'RGN_REF_OUTER', 'RGN_QRY_OUTER',
#     'FLAG_TYPE',
#     'ALIGN_INDEX',
#     'CALL_SOURCE', 'VAR_SCORE',
#     'FILTER',
#     'SEQ'
#  ]


# def _call_inv_accept_flagged_region(row, allow_single_cluster=False, match_any=None):
#     """
#     Annotate which flagged regions are "accepted" (will try to call INV).
#
#     If `allow_single_cluster` is `False` and `match_any` is an empty set, then the only signatures accepted are
#     matched SV events or matched indel events.
#
#     :param row: Row from merged flagged regions.
#     :param allow_single_cluster: Try to resolve inversions if True for loci with only signatures of clustered SNVs
#         and/or clustered indels. Will try many regions and may increase false-positives.
#     :param match_any: If defined, contains a set of signatures where at least one must match. Can be used to
#         restrict accepted regions to SV-supported signatures only, but inversions with a small uniquely-inverted
#         region will likely be missed.
#
#     :return: `True` if the inversion caller should try the region.
#     """
#
#     if match_any is None:
#         match_any = set()
#
#     if not allow_single_cluster and (row['TYPE'] == {'CLUSTER_SNV'} or row['TYPE'] == {'CLUSTER_INDEL'}):
#         return False
#
#     if match_any and not row['TYPE'] & match_any:
#         return False
#
#     return True
#
#
# #############
# ### Rules ###
# #############
#
#
# #
# # Call inversions
# #
#
# # Call all inversions from flagged loci
# rule call_inv_all:
#     input:
#         bed=lambda wildcards: pavcall.pipeline.expand_pattern(
#             'temp/{asm_name}/inv_caller/sv_inv_{hap}.bed.gz', ASM_TABLE, config
#         )
#
# # Call all inversions from variant call signatures
# rule call_inv_sig_all:
#     input:
#         bed=lambda wildcards: pavcall.pipeline.expand_pattern(
#             'temp/{asm_name}/inv_caller/sv_inv_{hap}.bed.gz', ASM_TABLE, config
#         )
#
#
# # Gather partitions.
# # noinspection PyTypeChecker
# rule call_inv_gather:
#     # noinspection PyUnresolvedReferences
#     input:
#         bed=lambda wildcards: [
#             f'temp/{wildcards.asm_name}/inv_caller/part/{wildcards.hap}/inv_call_{part}-of-{part_count}.bed.gz'
#                 for part in range(get_config('inv_sig_part_count', wildcards)) for part_count in (get_config('inv_sig_part_count', wildcards),)
#         ]
#     output:
#         bed=temp('temp/{asm_name}/inv_caller/sv_inv_{hap}.bed.gz')
#     params:
#         inv_max_overlap=lambda wildcards: get_config('inv_max_overlap', wildcards)
#     run:
#
#         df = pd.concat(
#             [pd.read_csv(file_name, sep='\t') for file_name in input.bed],
#             axis=0
#         ).sort_values(
#             ['VAR_SCORE']
#         )
#
#         # Drop overlapping variants (may be discovered from two separate flagged sites)
#         if df.shape[0] > 0:
#             inv_tree = collections.defaultdict(intervaltree.IntervalTree)
#
#             df_list = list()
#
#             for index, row in df.iterrows():
#                 keep = True
#
#                 for interval in inv_tree[row['#CHROM']].overlap(row['POS'], row['END']):
#                     if svpoplib.variant.reciprocal_overlap(row['POS'], row['END'], interval.begin, interval.end) >= params.inv_max_overlap:
#                         keep = False
#                         break
#
#                 if keep:
#                     df_list.append(row)
#
#             df = pd.concat(df_list, axis=1).T
#
#         df['ID'] = svpoplib.variant.get_variant_id(df)
#
#         df.sort_values(
#             ['#CHROM', 'POS', 'END', 'ID']
#         ).to_csv(
#             output.bed, sep='\t', index=False, compression='gzip'
#         )
#
#
# Call inversions in partitnions of flagged regions.
rule call_inv:
    input:
        pq_flag='results/{asm_name}/call/inv_flagged_regions_{hap}.parquet',
        pq_align='results/{asm_name}/align/{hap}/align_trim-qry.parquet',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
        ref_fofn='data/ref/ref.fofn',
    output:
        pq=temp('temp/{asm_name}/call/intra_inv_{hap}.parquet')
    log:
        log='log/{asm_name}/call/intra_inv_{hap}.log.gz'
    threads: 1
    run:

        pav_params = pavcall.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE)

        df_flag = (
            pl.scan_parquet(input.pq_flag)
            .filter(pl.col('flag') != ['CLUSTER_SNV'])  # Ignore SNV-only clusters
            .collect()
        )

        k_util = agglovar.kmer.util.KmerUtil(pav_params.inv_k_size)

        align_lift = pavcall.align.lift.AlignLift(
            pl.read_parquet(input.pq_align),
            agglovar.fa.read_fai(pavcall.pipeline.expand_fofn(input.qry_fofn)[1], name='qry_id')
        )

        kde_model = pavcall.kde.KdeTruncNorm(
            pav_params.inv_kde_bandwidth, pav_params.inv_kde_trunc_z, pav_params.inv_kde_func
        )

        ref_fa_filename, ref_fai_filename = pavcall.pipeline.expand_fofn(input.ref_fofn)[:2]
        qry_fa_filename, qry_fai_filename = pavcall.pipeline.expand_fofn(input.qry_fofn)[:2]

        # Call inversions
        call_list = list()

        with gzip.open(log.log, 'wt') as log_file:
            log_file.write(f'Scanning {df_flag.height} flagged regions for inversions\n')

            for row in df_flag.iter_rows(named=True):

                # Scan for inversions
                region_flag = pavcall.region.Region(row['chrom'], row['pos'], row['end'])

                log_file.write(
                    f'Intra-inversion flagged region: {row["chrom"]}:{row["pos"]}-{row["end"]} '
                    f'(flags="{", ".join(row["flag"])}", clusters={row["clusters"]})\n'
                )

                try:
                    inv_call = pavcall.inv.scan_for_inv(
                        region_flag=region_flag,
                        ref_fa_filename=(ref_fa_filename, ref_fai_filename),
                        qry_fa_filename=(qry_fa_filename, qry_fai_filename),
                        align_lift=align_lift,
                        pav_params=pav_params,
                        k_util=k_util,
                        kde_model=kde_model,
                        log_file=log_file
                    )

                except RuntimeError as ex:
                    log_file.write('RuntimeError in scan_for_inv(): {}\n'.format(ex))
                    inv_call = None

                # Save inversion call
                if inv_call is not None:
                    if pav_params.debug:
                        print(f'Found inversion: {inv_call}')

                    # Get seq
                    seq = pavlib.seq.region_seq_fasta(
                        inv_call.region_qry_outer,
                        input.fa_qry,
                        rev_compl=inv_call.region_qry_outer.is_rev
                    )

                    # Get alignment record data
                    align_index = ','.join(sorted(
                        pavlib.util.collapse_to_set(
                            (
                                inv_call.region_ref.pos_aln_index,
                                inv_call.region_ref.end_aln_index,
                                inv_call.region_ref_outer.pos_aln_index,
                                inv_call.region_ref_outer.end_aln_index
                            ),
                            to_type=str
                        )
                    ))

                    # Save call
                    call_list.append(pd.Series(
                        [
                            inv_call.region_ref.chrom,
                            inv_call.region_ref.pos,
                            inv_call.region_ref.end,

                            inv_call.id,
                            'INV',
                            inv_call.svlen,

                            wildcards.hap,

                            inv_call.region_qry.chrom, inv_call.region_qry.pos, inv_call.region_qry.end,
                            '-' if inv_call.region_qry_outer.is_rev else '+',

                            0,

                            inv_call.region_ref_outer.to_base1_string(),
                            inv_call.region_qry_outer.to_base1_string(),

                            row['TYPE'],

                            align_index,

                            call_source,
                            inv_call.score,

                            'PASS',

                            seq
                        ],
                        index=[
                            '#CHROM', 'POS', 'END',
                            'ID', 'SVTYPE', 'SVLEN',
                            'HAP',
                            'QRY_ID', 'QRY_POS', 'QRY_END', 'QRY_STRAND',
                            'CI',
                            'RGN_REF_OUTER', 'RGN_QRY_OUTER',
                            'FLAG_TYPE',
                            'ALIGN_INDEX',
                            'CALL_SOURCE', 'VAR_SCORE',
                            'FILTER',
                            'SEQ'
                        ]
                    ))

        # Merge records
        if len(call_list) > 0:
            df_bed = pd.concat(call_list, axis=1).T.sort_values(['#CHROM', 'POS', 'END', 'ID'])

        else:
            # Create emtpy data frame
            df_bed = pd.DataFrame(
                [],
                columns=INV_CALL_COLUMNS
            )

        # Write
        df_bed.to_csv(output.bed, sep='\t', index=False, compression='gzip')


# Flag regions where alignments may cross inversions without being split into multiple aligned segments.
rule call_inv_intra_flag:
    input:
        pq_snv='temp/{asm_name}/call/intra_snv_{hap}.parquet',
        pq_insdel='temp/{asm_name}/call/intra_insdel_{hap}.parquet',
        qry_fofn='data/query/{asm_name}/query_{hap}.fofn',
        ref_fofn='data/ref/ref.fofn',
    output:
        pq='results/{asm_name}/call/inv_flagged_regions_{hap}.parquet'
    run:

        pavcall.call.inv.cluster_merge(
            df_snv = pl.scan_parquet(input.pq_snv),
            df_insdel = pl.scan_parquet(input.pq_insdel),
            df_ref_fai = agglovar.fa.read_fai(pavcall.pipeline.expand_fofn(input.ref_fofn)[1], name='chrom').lazy(),
            df_qry_fai = agglovar.fa.read_fai(pavcall.pipeline.expand_fofn(input.qry_fofn)[1], name='qry_id').lazy(),
            pav_params=pavcall.params.PavParams(wildcards.asm_name, PAV_CONFIG, ASM_TABLE),
        ).write_parquet(output.pq)
