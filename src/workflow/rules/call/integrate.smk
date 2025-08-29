#
# #
# # Merge haplotypes
# #
#
# # Generate all BED files
# localrules: call_all_bed
#
# rule call_all_bed:
#     input:
#         bed_pass=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'results/{asm_name}/bed_merged/{filter}/{vartype_svtype}.bed.gz', ASM_TABLE, config,
#             vartype_svtype=('svindel_ins', 'svindel_del', 'sv_inv', 'snv_snv'), filter=('pass',)
#         ),
#         bed_fail=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'results/{asm_name}/bed_merged/{filter}/{vartype_svtype}.bed.gz', ASM_TABLE, config,
#             vartype_svtype=('svindel_ins', 'svindel_del', 'sv_inv', 'snv_snv'), filter=('fail',)
#         )
#
#
#
# # Concatenate variant BED files from batched merges - non-SNV (has variant FASTA).
# # noinspection PyTypeChecker
# rule call_merge_haplotypes:
#     input:
#         bed_batch=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'temp/{asm_name}/bed/merge_hap/{filter}/{vartype_svtype}/{part}-of-{part_count}.bed.gz', ASM_TABLE, config,
#             part=range(get_config("merge_partitions", wildcards)), part_count=(get_config("merge_partitions", wildcards),),
#             filter=wildcards.filter, vartype_svtype=wildcards.vartype_svtype
#         )
#     output:
#         bed='results/{asm_name}/bed_merged/{filter}/{vartype_svtype}.bed.gz',
#         fa='results/{asm_name}/bed_merged/{filter}/fa/{vartype_svtype}.fa.gz'
#     wildcard_constraints:
#         filter='pass|fail',
#         vartype_svtype='svindel_ins|svindel_del|sv_inv|sv_cpx'
#     run:
#
#         df_list = [pd.read_csv(file_name, sep='\t') for file_name in input.bed_batch if os.stat(file_name).st_size > 0]
#
#         df = pd.concat(
#             df_list, axis=0
#         ).sort_values(
#             ['#CHROM', 'POS', 'END', 'ID']
#         )
#
#         with Bio.bgzf.BgzfWriter(output.fa, 'wb') as out_file:
#             Bio.SeqIO.write(svpoplib.seq.bed_to_seqrecord_iter(df), out_file, 'fasta')
#
#         del df['SEQ']
#
#         df.to_csv(
#             output.bed, sep='\t', index=False, compression='gzip'
#         )
#
# # Concatenate variant BED files from batched merges - SNV (no variant FASTA).
# rule call_merge_haplotypes_snv:
#     input:
#         bed_batch=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'temp/{asm_name}/bed/merge_hap/{filter}/snv_snv/{part}-of-{part_count}.bed.gz', ASM_TABLE, config,
#             part=range(get_config("merge_partitions", wildcards)), part_count=(get_config("merge_partitions", wildcards),),
#             filter=wildcards.filter
#         )
#     output:
#         bed='results/{asm_name}/bed_merged/{filter}/snv_snv.bed.gz'
#     wildcard_constraints:
#         filter='pass|fail'
#     run:
#
#         # noinspection PyTypeChecker
#         df_list = [pd.read_csv(file_name, sep='\t') for file_name in input.bed_batch if os.stat(file_name).st_size > 0]
#
#         df = pd.concat(
#             df_list, axis=0
#         ).sort_values(
#             ['#CHROM', 'POS', 'END', 'ID']
#         ).to_csv(
#             output.bed, sep='\t', index=False, compression='gzip'
#         )
#
# # Merge by batches.
# rule call_merge_haplotypes_batch:
#     input:
#         tsv_part=lambda wildcards: f'data/ref/partition_{get_config('merge_partitions', wildcards)}.tsv.gz',
#         bed_var=lambda wildcards: pavlib.pipeline.expand_pattern(
#             (
#                 'results/{asm_name}/bed_hap/{hap}/pass_{vartype_svtype}.bed.parquet'
#                     if wildcards.filter == 'pass' else
#                         'results/{asm_name}/bed_hap/{hap}/fail/fail_{vartype_svtype}.bed.parquet'
#             ), ASM_TABLE, config,
#             asm_name=wildcards.asm_name, vartype_svtype=wildcards.vartype_svtype, hap=pavlib.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE)
#         ),
#         bed_callable=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'results/{asm_name}/callable/callable_regions_{hap}_500.bed.gz', ASM_TABLE, config,
#                 asm_name=wildcards.asm_name, hap=pavlib.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE)
#         )
#     output:
#         bed='temp/{asm_name}/bed/merge_hap/{filter}/{vartype_svtype}/{part}-of-{part_count}.bed.gz'
#     wildcard_constraints:
#         filter='pass|fail',
#         vartype_svtype='svindel_ins|svindel_del|sv_inv|sv_cpx|snv_snv'
#     threads: 12
#     run:
#
#         pav_params = pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE)
#
#         config_def = pavlib.call.get_merge_params(wildcards.vartype_svtype.split('_')[1], pav_params)
#
#         variant_pattern = (
#             'results/{asm_name}/bed_hap/{hap}/pass_{vartype_svtype}.bed.parquet'
#                 if wildcards.filter == 'pass' else
#                     'results/{asm_name}/bed_hap/{hap}/fail/fail_{vartype_svtype}.bed.parquet'
#         )
#
#         callable_pattern = 'results/{asm_name}/callable/callable_regions_{hap}_500.bed.gz'
#
#         hap_list = pavlib.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE)
#
#         is_cpx = wildcards.vartype_svtype == 'sv_cpx'
#
#         # Get a list of chromosomes in this batch
#         chrom_list = pl.scan_csv(
#             input.tsv_part, separator='\t'
#         ).filter(
#             pl.col('PARTITION') == int(wildcards.part)
#         ).select(
#             pl.col('CHROM').unique()
#         ).collect().to_series().sort().to_list()
#
#         # Read batch table
#         bed_list = [
#             pl.scan_parquet(
#                 variant_pattern.format(hap=hap, **wildcards)
#             ).filter(
#                 pl.col('#CHROM').is_in(
#                     chrom_list
#                 )
#             ).collect(
#             ).to_pandas(
#             ).set_index(
#                 'ID', drop=False
#             )
#                 for hap in hap_list
#         ]
#
#         for df in bed_list:
#             df.index.name = 'INDEX'
#
#         # Transform CPX
#         if is_cpx:
#             for df in bed_list:
#                 df['_END_ORG'] = df['END']
#                 df['END'] = df['POS'] + df['SVLEN']
#
#         # Get configured merge definition
#         print('Merging with def: ' + config_def)
#         sys.stdout.flush()
#
#         # Merge
#         df = pavlib.call.merge_haplotypes(
#             bed_list,
#             input.bed_callable,
#             hap_list,
#             config_def,
#             threads=threads
#         )
#
#         # Restore CPX transformation
#         if is_cpx:
#             for df in bed_list:
#                 df['END'] = df['_END_ORG']
#                 del df['_END_ORG']
#
#         # Write
#         df.to_csv(output.bed, sep='\t', index=False, compression='gzip')
#
#
# #
# # Merge support
# #
#
# localrules: call_callable_regions_all
#
# # Generate BED files for all callable regions
# rule call_callable_regions_all:
#     input:
#         bed=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'results/{asm_name}/callable/callable_regions_{hap}_500.bed.gz', ASM_TABLE, config
#         )
#
#
# # Make a table of mappable regions by merging aligned loci with loci covered by alignment-truncating events.
# # "flank" parameter is an integer describing how far away records may be to merge (similar to the "bedtools merge"
# # "slop" parameter). The flank is not added to the regions that are output.
# rule call_callable_regions:
#     input:
#         bed_align='results/{asm_name}/align/trim-qryref/align_qry_{hap}.bed.gz',
#         bed_lg_del='results/{asm_name}/lgsv/svindel_del_{hap}.bed.gz',
#         bed_lg_ins='results/{asm_name}/lgsv/svindel_ins_{hap}.bed.gz',
#         bed_lg_inv='results/{asm_name}/lgsv/sv_inv_{hap}.bed.gz',
#         bed_lg_cpx='results/{asm_name}/lgsv/sv_cpx_{hap}.bed.gz'
#     output:
#         bed='results/{asm_name}/callable/callable_regions_{hap}_{flank}.bed.gz'
#     run:
#
#         # Get flank param
#         try:
#             flank = int(wildcards.flank)
#
#         except ValueError:
#             raise RuntimeError('Flank parameter is not an integer: {flank}'.format(**wildcards))
#
#         # Merge
#         df = pavlib.util.region_merge(
#             [
#                 input.bed_align,
#                 input.bed_lg_del,
#                 input.bed_lg_ins,
#                 input.bed_lg_inv,
#                 input.bed_lg_cpx
#             ],
#             pad=flank
#         )
#
#         # Write
#         df.to_csv(output.bed, sep='\t', index=False, compression='gzip')
#
#
# #
# # Integrate variant calls from multiple sources (per haplotype, pre-merge)
# #
#
# # Run all BED
# localrules: call_all_bed_hap
#
# rule call_all_bed_hap:
#     input:
#         bed_pass=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'results/{asm_name}/bed_hap/{hap}/pass_{vartype_svtype}.bed.{ext}', ASM_TABLE, config,
#             vartype_svtype=('svindel_ins', 'svindel_del', 'sv_inv', 'snv_snv', 'sv_cpx'), ext=('parquet', 'gz')
#         ),
#         bed_fail=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'results/{asm_name}/bed_hap/{hap}/fail/fail_{vartype_svtype}.bed.{ext}', ASM_TABLE, config,
#             vartype_svtype=('svindel_ins', 'svindel_del', 'sv_inv', 'snv_snv'), ext=('parquet', 'gz')
#         )
#
# # Create PASS BEDs
# localrules: call_all_bed_hap_pass
#
# rule call_all_bed_hap_pass:
#     input:
#         bed_pass=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'results/{asm_name}/bed_hap/{hap}/pass_{vartype_svtype}.bed.parquet', ASM_TABLE, config,
#             vartype_svtype=('svindel_ins', 'svindel_del', 'sv_inv', 'snv_snv', 'sv_cpx')
#         )
#
# # Remove redundant variants from the FAIL BED.
# rule call_fail_redundant:
#     input:
#         bed_pass_ins='results/{asm_name}/bed_hap/{hap}/pass_svindel_ins.bed.parquet',
#         bed_fail_ins='temp/{asm_name}/bed_hap/{hap}/fail_svindel_ins.bed.parquet',
#         bed_pass_del='results/{asm_name}/bed_hap/{hap}/pass_svindel_del.bed.parquet',
#         bed_fail_del='temp/{asm_name}/bed_hap/{hap}/fail_svindel_del.bed.parquet',
#         bed_pass_inv='results/{asm_name}/bed_hap/{hap}/pass_sv_inv.bed.parquet',
#         bed_fail_inv='temp/{asm_name}/bed_hap/{hap}/fail_sv_inv.bed.parquet',
#         bed_align='results/{asm_name}/align/trim-none/align_qry_{hap}.bed.gz',
#     output:
#         bed_fail_ins='results/{asm_name}/bed_hap/{hap}/fail/fail_svindel_ins.bed.parquet',
#         bed_drop_ins='results/{asm_name}/bed_hap/{hap}/dropped/redundanttrim_svindel_ins.bed.parquet',
#         bed_fail_del='results/{asm_name}/bed_hap/{hap}/fail/fail_svindel_del.bed.parquet',
#         bed_drop_del='results/{asm_name}/bed_hap/{hap}/dropped/redundanttrim_svindel_del.bed.parquet',
#         bed_fail_inv='results/{asm_name}/bed_hap/{hap}/fail/fail_sv_inv.bed.parquet',
#         bed_drop_inv='results/{asm_name}/bed_hap/{hap}/dropped/redundanttrim_sv_inv.bed.parquet'
#     params:
#         ro_threshold_insdel=0.5,
#         ro_threshold_inv=0.2
#     run:
#
#         pav_params = pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE)
#
#         # Read alignment scores
#         df_align_score = pl.scan_csv(
#             input.bed_align, separator='\t',
#             schema_overrides={
#                 'INDEX': pl.UInt32,
#                 'SCORE': pl.Float32
#             }
#         ).select(
#             pl.col.INDEX,
#             pl.col.SCORE.alias('_SCORE')  # Is added to variant frames by join, avoid name conflicts
#         ).collect()
#
#         for svtype in ('ins', 'del', 'inv'):
#
#             in_pass_filename = input[f'bed_pass_{svtype}']
#             in_fail_filename = input[f'bed_fail_{svtype}']
#             out_fail_filename = output[f'bed_fail_{svtype}']
#             out_drop_filename = output[f'bed_drop_{svtype}']
#
#             # If True, END is POS + SVLEN, otherwise, end is unaltered
#             is_ins_end = svtype == 'ins'
#
#             # Set RO threshold
#             if svtype == 'inv':
#                 ro_threshold = params.ro_threshold_inv
#             else:
#                 ro_threshold = params.ro_threshold_insdel
#
#             # Get a list of chromosomes
#             chroms = pl.scan_parquet(
#                 in_fail_filename
#             ).select(
#                 pl.col('#CHROM').unique()
#             ).collect().to_series().sort()
#
#             # Process each chromosome
#             drop_index_list = list()
#
#             for chrom in chroms:
#                 if pav_params.debug:
#                     print(f'Processing {chrom}')
#
#                 # Scan PASS variants
#                 df_pass = pl.scan_parquet(
#                     in_pass_filename
#                 ).select(
#                     pl.col('#CHROM'),
#                     pl.col('POS'),
#                     ((pl.col('POS') + pl.col('SVLEN')) if is_ins_end else pl.col('END')).alias('_END_RO'),
#                     pl.col('SVLEN')
#                 ).filter(
#                     pl.col('#CHROM') == chrom
#                 ).drop(pl.col('#CHROM'))
#
#                 # Scan FAIL variants
#                 df_fail = pl.scan_parquet(
#                     in_fail_filename
#                 ).select(
#                     pl.col('#CHROM'),
#                     pl.col('POS'),
#                     pl.col('END'),
#                     pl.col('SVLEN'),
#                     pl.col('FILTER'),
#                     pl.col('ALIGN_INDEX')
#                 ).with_row_index(
#                     '_index'
#                 ).filter(
#                     pl.col('#CHROM') == chrom
#                 ).drop(
#                     pl.col('#CHROM')
#                 ).with_columns(
#                     ((pl.col('POS') + pl.col('SVLEN')) if is_ins_end else pl.col('END')).alias('_END_RO'),
#                 )
#
#                 # Split by TRIM-only filter (filter TRIM-only by overlaps, retain all non-TRIM-only)
#                 df_fail = df_fail.with_columns(
#                     _IS_TRIM=(pl.col.FILTER.str.split(',').list.set_difference(['TRIMREF', 'TRIMQRY']).list.len() == 0).cast(pl.Boolean)
#                 )
#
#                 # Add non-TRIM-only records to PASS variants
#                 df_pass = pl.concat([
#                     df_pass,
#                     df_fail.filter(
#                         ~pl.col._IS_TRIM
#                     ).select(
#                         pl.col('POS'), pl.col('_END_RO'), pl.col('SVLEN')
#                     )
#                 ]).sort(
#                     pl.col('POS'), pl.col('_END_RO')
#                 )
#
#                 df_fail = df_fail.filter(pl.col._IS_TRIM).drop('_IS_TRIM')
#
#                 # Get minimum alignment score for each FAIL record
#                 df_fail = df_fail.with_columns(
#                     _ALIGN_INDEX=pl.col.ALIGN_INDEX.str.split(',').cast(pl.List(pl.UInt32))
#                 ).explode(
#                     pl.col._ALIGN_INDEX
#                 ).join(
#                     df_align_score.lazy(), left_on='_ALIGN_INDEX', right_on='INDEX', how='left'
#                 ).filter(
#                     pl.all_horizontal(
#                         pl.col._SCORE == pl.col._SCORE.min(),
#                         pl.col._SCORE.is_first_distinct()
#                     ).over('_index')
#                 ).drop(pl.col._ALIGN_INDEX)
#
#                 # Filter by self intersect. Get indices to drop.
#                 # Remove any records in the FAIL variants that overlap with another record in FAIL variants by RO threshold.
#                 # For overlaps, prioritize by a higher alignment score (keep first if scores are equal).
#                 drop_index_fail = df_fail.select(
#                     pl.col._index, pl.col.POS, pl.col._END_RO, pl.col.SVLEN, pl.col._SCORE
#                 ).join_where(
#                     # Join overlapping records
#                     df_fail.select(
#                         pl.col._index, pl.col.POS, pl.col._END_RO, pl.col.SVLEN, pl.col._SCORE
#                     ),
#                     pl.col._index != pl.col._index_right,
#                     pl.col.POS < pl.col._END_RO_right,
#                     pl.col.POS_right < pl.col._END_RO
#                 ).filter(
#                     (
#                         # Filter by RO
#                         (
#                             pl.min_horizontal([pl.col._END_RO, pl.col._END_RO_right]) - pl.max_horizontal([pl.col.POS, pl.col.POS_right])
#                         ) / (
#                             pl.max_horizontal([pl.col.SVLEN, pl.col.SVLEN_right])
#                         ) >= ro_threshold
#                     ) & (
#                         # Drop lower score; drop higher index if scores are equal (keep first)
#                         (pl.col._SCORE > pl.col._SCORE_right) | (
#                             (pl.col._SCORE == pl.col._SCORE_right) & (pl.col._index < pl.col._index_right)
#                         )
#                     )
#                 ).select(
#                     pl.col._index_right
#                 )
#
#                 # Filter by PASS intersect. Get indices to drop.
#                 drop_index_pass = df_fail.join_where(
#                     df_pass,
#                     pl.col.POS < pl.col._END_RO_right,
#                     pl.col.POS_right < pl.col._END_RO
#                 ).filter(
#                     (
#                         pl.min_horizontal([pl.col._END_RO, pl.col._END_RO_right]) - pl.max_horizontal([pl.col.POS, pl.col.POS_right])
#                     ) / (
#                         pl.max_horizontal([pl.col.SVLEN, pl.col.SVLEN_right])
#                     ) >= ro_threshold
#                 ).select(
#                     pl.col._index
#                 )
#
#                 # Collect and retain dropped indices
#                 drop_index_list.append(
#                     drop_index_fail.collect(engine='streaming').to_series()
#                 )
#
#                 drop_index_list.append(
#                     drop_index_pass.collect(engine='streaming').to_series()
#                 )
#
#             # Concatenate dropped indices
#             if not drop_index_list:
#                 drop_index_list = [pl.Series([], dtype=pl.UInt32)]  # List of one empty series for concatenation
#
#             drop_index = pl.concat(
#                 drop_index_list
#             ).unique().sort().to_list()
#
#             # Write output BED files
#             df_fail = pl.scan_parquet(
#                 in_fail_filename
#             ).with_row_index(
#                 '_index'
#             )
#
#             df_fail_retain = df_fail.filter(
#                 ~pl.col._index.is_in(drop_index)
#             ).drop(
#                 pl.col._index
#             )
#
#             df_fail_drop = df_fail.filter(
#                 pl.col._index.is_in(drop_index)
#             ).drop(
#                 pl.col._index
#             )
#
#             out_bed_retain = df_fail_retain.sink_parquet(
#                 out_fail_filename,
#                 lazy=True
#             )
#
#             out_bed_drop = df_fail_drop.sink_parquet(
#                 out_drop_filename,
#                 lazy=True
#             )
#
#             pl.collect_all([out_bed_retain, out_bed_drop])
#
# rule call_fail_redundant_snv:
#     input:
#         bed_pass='results/{asm_name}/bed_hap/{hap}/pass_snv_snv.bed.parquet',
#         bed_fail='temp/{asm_name}/bed_hap/{hap}/fail_snv_snv.bed.parquet',
#         bed_align='results/{asm_name}/align/trim-none/align_qry_{hap}.bed.gz',
#     output:
#         bed_fail='results/{asm_name}/bed_hap/{hap}/fail/fail_snv_snv.bed.parquet',
#         bed_drop='results/{asm_name}/bed_hap/{hap}/dropped/redundanttrim_snv_snv.bed.parquet'
#     run:
#
#         pav_params = pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE)
#
#         # Read alignment scores
#         df_align_score = pl.scan_csv(
#             input.bed_align, separator='\t',
#             schema_overrides={
#                 'INDEX': pl.UInt32,
#                 'SCORE': pl.Float32
#             }
#         ).select(
#             pl.col.INDEX,
#             pl.col.SCORE.alias('_SCORE')  # Is added to variant frames by join, avoid name conflicts
#         ).collect()
#
#         # Get a list of chromosomes
#         chroms = pl.scan_parquet(
#             input.bed_fail
#         ).select(
#             pl.col('#CHROM').unique()
#         ).collect().to_series().sort()
#
#         # Process each chromosome
#         drop_index_list = list()
#
#         for chrom in chroms:
#             if pav_params.debug:
#                 print(f'Processing {chrom}')
#
#             # Scan PASS variants
#             df_pass = pl.scan_parquet(
#                 input.bed_pass,
#             ).select(
#                 pl.col('#CHROM'),
#                 pl.col('POS'),
#                 pl.col('ALT').str.to_uppercase(),
#             ).filter(
#                 pl.col('#CHROM') == chrom
#             ).drop(pl.col('#CHROM'))
#
#             # Scan FAIL variants
#             df_fail = pl.scan_parquet(
#                 input.bed_fail
#             ).select(
#                 pl.col('#CHROM'),
#                 pl.col('POS'),
#                 pl.col('ALT').str.to_uppercase(),
#                 pl.col('FILTER'),
#                 pl.col('ALIGN_INDEX')
#             ).with_row_index(
#                 '_index'
#             ).filter(
#                 pl.col('#CHROM') == chrom
#             ).drop(
#                 pl.col('#CHROM')
#             )
#
#             # Split by TRIM-only filter (filter TRIM-only by overlaps, retain all non-TRIM-only)
#             df_fail = df_fail.with_columns(
#                 _IS_TRIM=(pl.col.FILTER.str.split(',').list.set_difference(['TRIMREF', 'TRIMQRY']).list.len() == 0).cast(pl.Boolean)
#             ).drop('FILTER')
#
#             # Add non-TRIM-only records to PASS variants
#             df_pass = pl.concat([
#                 df_pass,
#                 df_fail.filter(
#                     ~pl.col._IS_TRIM
#                 ).select(
#                     pl.col('POS'), pl.col('ALT')
#                 )
#             ]).sort(
#                 pl.col('POS'), pl.col('ALT')
#             )
#
#             df_fail = df_fail.filter(pl.col._IS_TRIM).drop('_IS_TRIM')
#
#             # Get minimum alignment score for each FAIL record
#             df_fail = df_fail.with_columns(
#                 ALIGN_INDEX=pl.col.ALIGN_INDEX.str.split(',').cast(pl.List(pl.UInt32))
#             ).explode(
#                 pl.col.ALIGN_INDEX
#             ).join(
#                 df_align_score.lazy(), left_on='ALIGN_INDEX', right_on='INDEX', how='left'
#             ).filter(
#                 pl.all_horizontal(
#                     pl.col._SCORE == pl.col._SCORE.min(),
#                     pl.col._SCORE.is_first_distinct()
#                 ).over('_index')
#             ).drop(pl.col.ALIGN_INDEX)
#
#             # Filter by self intersect. Get indices to drop.
#             drop_index_fail = df_fail.select(
#                 pl.col._index, pl.col.POS, pl.col.ALT, pl.col._SCORE
#             ).join_where(
#                 # Join overlapping records
#                 df_fail.select(
#                     pl.col._index, pl.col.POS, pl.col.ALT, pl.col._SCORE
#                 ),
#                 pl.col._index != pl.col._index_right,
#                 pl.col.POS == pl.col.POS_right,
#                 pl.col.ALT == pl.col.ALT_right
#             ).filter(
#                 # Drop lower score; drop higher index if scores are equal (keep first)
#                 (pl.col._SCORE > pl.col._SCORE_right) | (
#                     (pl.col._SCORE == pl.col._SCORE_right) & (pl.col._index < pl.col._index_right)
#                 )
#             ).select(
#                 pl.col._index_right
#             )
#
#             # Filter by PASS intersect. Get indices to drop.
#             drop_index_pass = df_fail.join(
#                 df_pass,
#                 on=[pl.col.POS, pl.col.ALT],
#                 how='semi'
#             ).select(
#                 pl.col._index
#             )
#
#             # Collect and retain dropped indices
#             drop_index_list.append(
#                 drop_index_fail.collect(engine='streaming').to_series()
#             )
#
#             drop_index_list.append(
#                 drop_index_pass.collect(engine='streaming').to_series()
#             )
#
#         # Concatenate dropped indices
#         if not drop_index_list:
#             drop_index_list = [pl.Series([], dtype=pl.UInt32)]  # List of one empty series for concatenation
#
#         drop_index = pl.concat(
#             drop_index_list
#         ).unique().sort().to_list()
#
#         # Write output BED files
#         df_fail = pl.scan_parquet(
#             input.bed_fail
#         ).with_row_index(
#             '_index'
#         )
#
#         df_fail_retain = df_fail.filter(
#             ~pl.col._index.is_in(drop_index)
#         ).drop(
#             pl.col._index
#         )
#
#         df_fail_drop = df_fail.filter(
#             pl.col._index.is_in(drop_index)
#         ).drop(
#             pl.col._index
#         )
#
#         out_bed_retain = df_fail_retain.sink_parquet(
#             output.bed_fail,
#             lazy=True
#         )
#
#         out_bed_drop = df_fail_drop.sink_parquet(
#             output.bed_drop,
#             lazy=True
#         )
#
#         pl.collect_all([out_bed_retain, out_bed_drop])
#
# localrules: call_integrate_all
#
# rule call_integrate_all:
#     input:
#         bed=lambda wildcards: pavlib.pipeline.expand_pattern(
#             'results/{asm_name}/bed_hap/{hap}/pass_svindel_ins.bed.parquet', ASM_TABLE, config
#         )
#
# # Filter variants from inside inversions
# rule call_integrate_sources:
#     input:
#         bed_cigar_insdel='temp/{asm_name}/cigar/svindel_insdel_{hap}.bed.gz',
#         bed_cigar_snv='temp/{asm_name}/cigar/snv_snv_{hap}.bed.gz',
#         bed_lg_ins='results/{asm_name}/lgsv/svindel_ins_{hap}.bed.gz',
#         bed_lg_del='results/{asm_name}/lgsv/svindel_del_{hap}.bed.gz',
#         bed_lg_inv='results/{asm_name}/lgsv/sv_inv_{hap}.bed.gz',
#         bed_lg_cpx='results/{asm_name}/lgsv/sv_cpx_{hap}.bed.gz',
#         bed_seg='results/{asm_name}/lgsv/segment_{hap}.bed.gz',
#         bed_inv='temp/{asm_name}/inv_caller/sv_inv_{hap}.bed.gz',
#         bed_depth_qry='results/{asm_name}/align/trim-qry/depth_ref_{hap}.bed.gz',
#         bed_align_none='results/{asm_name}/align/trim-none/align_qry_{hap}.bed.gz',
#         bed_align_qry='results/{asm_name}/align/trim-qry/align_qry_{hap}.bed.gz',
#         bed_align_qryref='results/{asm_name}/align/trim-qryref/align_qry_{hap}.bed.gz',
#         qry_fa='data/query/{asm_name}/query_{hap}.fa.gz'
#     output:
#         bed_ins_pass='results/{asm_name}/bed_hap/{hap}/pass_svindel_ins.bed.parquet',
#         bed_del_pass='results/{asm_name}/bed_hap/{hap}/pass_svindel_del.bed.parquet',
#         bed_inv_pass='results/{asm_name}/bed_hap/{hap}/pass_sv_inv.bed.parquet',
#         bed_cpx_pass='results/{asm_name}/bed_hap/{hap}/pass_sv_cpx.bed.parquet',
#         bed_snv_pass='results/{asm_name}/bed_hap/{hap}/pass_snv_snv.bed.parquet',
#         bed_ins_fail=temp('temp/{asm_name}/bed_hap/{hap}/fail_svindel_ins.bed.parquet'),
#         bed_del_fail=temp('temp/{asm_name}/bed_hap/{hap}/fail_svindel_del.bed.parquet'),
#         bed_inv_fail=temp('temp/{asm_name}/bed_hap/{hap}/fail_sv_inv.bed.parquet'),
#         bed_cpx_fail='results/{asm_name}/bed_hap/{hap}/fail_sv_cpx.bed.parquet',
#         bed_snv_fail=temp('temp/{asm_name}/bed_hap/{hap}/fail_snv_snv.bed.parquet')
#     run:
#
#         pav_params = pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE)
#
#         inv_min = pav_params.inv_min
#         inv_max = pav_params.inv_max
#
#         # Read non-trimmed alignments (temp)
#         df_align_none = pd.read_csv(
#             input.bed_align_none, sep='\t', usecols=lambda col: col not in {'CIGAR'}
#         )
#
#         # Read trimmed regions (regionsa are tuples of alignment records and coordinates within the record).
#         # IntervalTree where coordinates are tuples - (index, pos):(index, end)
#         trim_tree = pavlib.call.read_trim_regions(
#             df_align_none,
#             pd.read_csv(input.bed_align_qry, sep='\t'),
#             pd.read_csv(input.bed_align_qryref, sep='\t')
#         )
#
#         # Whole alignment record filters
#         record_filter = dict(
#             df_align_none.loc[df_align_none['FILTER'] != 'PASS', 'FILTER'].str.split(',').apply(set)
#         )
#
#         del df_align_none
#
#         # Read query filter (if present)
#         # Dict keyed by assembly sequence ID (contig name) and each value is an interval tree of filtered coordinates.
#         qry_filter_tree = pavlib.call.read_filter_tree(pav_params.query_filter)
#
#         # Create large variant filter tree (for filtering small variants inside larger ones)
#         # Dict keyed by assembly sequence ID (contig name) and each value is an interval tree of purged coordinates.
#         purge_filter_tree = collections.defaultdict(intervaltree.IntervalTree)
#
#         # Get a dict of alignment records internal to larger variants and update the purge filter tree for large
#         # variants. For large variants where there is a gap between anchors, the gap is added to the purge filter tree.
#         # For large variants with overlapping anchors, the region that would be trimmed by reference trimming is added
#         # to the inner tree.
#         #
#         # Return dict:
#         # * Key: Alignment index.
#         # * Value: Tuple of the variant ID and a set of filters.
#         inner_tree = pavlib.call.read_inner_tree(input.bed_seg, trim_tree, purge_filter_tree)
#
#         # Tuple of (pass, drop) filenames for each variant type (variant pass, variant fail, fasta pass, fasta fail)
#         out_filename_dict = {
#             'ins': (output.bed_ins_pass, output.bed_ins_fail),
#             'del': (output.bed_del_pass, output.bed_del_fail),
#             'inv': (output.bed_inv_pass, output.bed_inv_fail),
#             'cpx': (output.bed_cpx_pass, output.bed_cpx_fail),
#             'snv': (output.bed_snv_pass, output.bed_snv_fail),
#             'lg_cpx': (output.bed_cpx_pass, output.bed_cpx_fail)
#         }
#
#         #
#         # Integrate and filter variants
#         #
#
#         df_insdel_list = list()  # Collect INS/DEL tables into this list until they are merged and written
#         df_inv_list = list()     # Collect INV tables into this list until they are merged and written
#         df_dup_list = list()     # Collect DUP loci
#
#         param_dict = { # do_write, is_insdel, is_inv, is_lg, add_purge, filter_purge, filter_inner, filename
#             'lg_cpx':  ( True,     False,     False,  True,  False,     False,        False,        input.bed_lg_cpx ),
#             'lg_ins':  ( False,    True,      False,  True,  False,     False,        False,        input.bed_lg_ins ),
#             'lg_del':  ( False,    True,      False,  True,  False,     False,        False,        input.bed_lg_del ),
#             'lg_inv':  ( False,    False,     True,   True,  False,     False,        False,        input.bed_lg_inv ),
#             'inv':     ( True,     False,     True,   False, True,      False,        True,         input.bed_inv ),
#             'insdel':  ( True,     True,      False,  False, False,     True,         True,         input.bed_cigar_insdel ),
#             'snv':     ( True,     False,     False,  False, False,     True,         True,         input.bed_cigar_snv )
#         }
#
#         # do_write: Write variant call table. Set to False for INS/DEL or INV variants until they are all collected
#         # is_insdel: Variant is an INS/DEL type, append to df_insdel_list
#         # is_inv: Variant is an inversion, apply inversion filtering steps and append to df_inv_list
#         # is_lg: Large variant (may have inner variants).
#         # add_purge: Add variant regions to the PURGE regions.
#         # filter_purge: Apply PURGE filter.
#         # filter_inner: Apply INNER filter.
#         # filename: Input variant filename.
#
#         for vartype in ('lg_cpx', 'lg_ins', 'lg_del', 'lg_inv', 'inv', 'insdel', 'snv'):
#             if pav_params.debug:
#                 print(f'Processing {vartype}')
#
#             if vartype not in param_dict:
#                 raise RuntimeError(f'vartype in control loop does not match a known value: {vartype}')
#
#             do_write, is_insdel, is_inv, is_lg, add_purge, filter_purge, filter_inner, filename = param_dict[vartype]
#
#             # Read
#             df, filter_dict = pavlib.call.read_variant_table(filename, True)
#
#             if len(set(df.columns) & {'QRY_ID', 'QRY_POS', 'QRY_END'}) != 3:
#                 raise RuntimeError(f'Missing QRY_ID, QRY_POS, and/or QRY_END columns: {filename}')
#
#             if 'SEQ' not in df.columns:
#                 df['SEQ'] = [str(seq.seq) for seq in pavlib.seq.variant_seq_from_region(df, input.qry_fa)]
#
#             # Set info dictionaries for filters
#             purge_dict = collections.defaultdict(set)
#             inner_dict = collections.defaultdict(set)
#
#             # Override purging
#             if pav_params.redundant_callset:
#                 filter_lgpruge = False
#                 add_purge = False
#
#             # Apply filters
#             if df.shape[0] > 0:
#
#                 # Apply query filtered regions
#                 pavlib.call.apply_qry_filter_tree(df, qry_filter_tree, filter_dict)
#
#                 if is_inv:
#
#                     # SVLEN min
#                     if inv_min is not None:
#                         for index in df.index[df['SVLEN'] < inv_min]:
#                             filter_dict[index].add('SVLEN')
#
#                     # SVLEN max
#                     if inv_max is not None and inv_max > 0:
#                         for index in df.index[df['SVLEN'] > inv_max]:
#                             filter_dict[index].add('SVLEN')
#
#                 # Filter PURGE
#                 if filter_purge:
#                     pavlib.call.apply_purge_filter(
#                         df=df,
#                         purge_filter_tree=purge_filter_tree,
#                         filter_dict=filter_dict,
#                         purge_dict=purge_dict,
#                         update=add_purge
#                     )
#
#                 elif add_purge:
#                     for index, row in df.iterrows():
#                         purge_filter_tree[row['#CHROM']][row['POS']:row['END']] = row['ID']
#
#                 # Filter TRIMREF & TRIMQRY
#                 if not (is_lg or is_inv):
#                     pavlib.call.apply_trim_filter(
#                         df=df,
#                         filter_dict=filter_dict,
#                         trim_tree=trim_tree
#                     )
#
#                 # Filter INNER
#                 if filter_inner:
#                     pavlib.call.apply_inner_filter(
#                         df=df,
#                         filter_dict=filter_dict,
#                         inner_tree=inner_tree,
#                         inner_dict=inner_dict
#                     )
#
#             # Update PURGE fields
#             pavlib.call.update_fields(df, filter_dict, purge_dict, inner_dict)
#
#             del filter_dict
#             del purge_dict
#             del inner_dict
#
#             # Version variant IDs prioritizing PASS over non-PASS (avoid version suffixes on PASS).
#             df['ID'] = pavlib.call.version_variant_bed_id(df)
#
#             # # Remove fields
#             # for col in ['QRY_ID', 'QRY_POS', 'QRY_END']:
#             #     if col in df.columns:
#             #         del df[col]
#
#             # Aggregate if type is split over multiple inputs
#             if is_insdel:
#                 df_insdel_list.append(df)
#
#             if is_inv:
#                 df_inv_list.append(df)
#
#             # Write
#             if do_write:
#
#                 if is_inv:
#                     df = pd.concat(df_inv_list, axis=0).sort_values(['#CHROM', 'POS'])
#
#                 if is_insdel:
#                     # Merge
#                     df = pd.concat(df_insdel_list, axis=0).sort_values(['#CHROM', 'POS'])
#
#                     del df_insdel_list
#
#                     # Separate INS and DEL, then write
#                     for svtype in ('ins', 'del'):
#
#                         filename_pass, filename_fail = out_filename_dict[svtype]
#
#                         # Pass
#                         pavlib.call.pandas_to_polars(
#                             df.loc[
#                                 (df['SVTYPE'] == svtype.upper()) & (df['FILTER'] == 'PASS')
#                             ]
#                         ).write_parquet(
#                             filename_pass
#                         )
#
#                         # Fail
#                         pavlib.call.pandas_to_polars(
#                             df.loc[
#                                 (df['SVTYPE'] == svtype.upper()) & (df['FILTER'] != 'PASS')
#                             ]
#                         ).write_parquet(
#                             filename_fail
#                         )
#
#                 else:
#                     filename_pass, filename_fail = out_filename_dict[vartype]
#
#                     # Pass
#                     pavlib.call.pandas_to_polars(
#                         df.loc[df['FILTER'] == 'PASS']
#                     ).write_parquet(
#                         filename_pass
#                     )
#
#                     # Fail
#                     pavlib.call.pandas_to_polars(
#                         df.loc[df['FILTER'] != 'PASS']
#                     ).write_parquet(
#                         filename_fail
#                     )
#
#             # Clean
#             del df
