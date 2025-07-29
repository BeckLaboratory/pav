"""
Prepare UCSC tracks for data.
"""

import matplotlib as mpl
import numpy as np
import os
import pandas as pd

import pavlib
import svpoplib

global ASM_TABLE
global PIPELINE_DIR
global REF_FAI
global temp


#
# Definitions
#

VARTYPE_TO_SVTYPE_TUPLE = {
    'snv': ('snv',),
    'sv': ('ins', 'del'),
    'indel': ('ins', 'del')
}

ALIGN_COLORMAP = 'viridis'

VAR_BED_PREMERGE_PATTERN = 'results/{{asm_name}}/bed_hap/{filter}/{{hap}}/{vartype}_{svtype}.bed.gz'

CPX_COLOR_DICT = {
    'INS': (0.2510, 0.2510, 1.0000),
    'DEL': (1.0000, 0.2510, 0.2510),
    'INV': (0.3765, 1.0000, 0.3765),

    'DUP': (1.0000, 0.2510, 1.0000),
    'TRP': (0.8510, 0.2118, 0.8510),
    'QUAD': (0.7020, 0.1765, 0.7020),
    'HDUP': (0.3765, 0.1255, 0.3765),

    'INVDUP': (0.3216, 0.8510, 0.3686),
    'INVTRP': (0.2745, 0.7020, 0.2745),
    'INVQUAD': (0.2157, 0.5490, 0.2157),
    'INVHDUP': (0.1569, 0.4000, 0.1569),

    'MIXDUP': (0.8510, 0.5216, 0.1451),
    'MIXTRP': (0.7020, 0.4275, 0.1176),
    'MIXQUAD': (0.5451, 0.3333, 0.0941),
    'MIXHDUP': (0.4000, 0.2471, 0.0706),

    'NML': (0.2000, 0.2000, 0.2000),

    'UNMAPPED': (0.4510, 0.4510, 0.4510)
}

def get_colors_pass_fail(color_dict, lighten_amt=0.25):
    """
    Get a dictionary of colors for PASS/FAIL records. Takes a dictionary of colors with keys describing the color
    (arbitrary string) and values of (R, G, B) tuples (0.0-1.0 values). Returns a dictionary with tuple keys with
    the original name and "True" for PASS and "False" for FAIL. For example, if the input dict has color "COLOR1",
    the output dict has keys ("COLOR1", True) and ("COLOR1", False) where "True" is the original color and "False"
    is an altered version of the color (lightened).

    :param color_dict: Input color dictionary.
    :param lighten_amt: Lightend FAIL coloors by this amount (0.0-1.0).

    :return: Output color dictionary with tuple keys ("COLOR", True) for PASS records and ("COLOR", False) for FAIL
        records for each "COLOR" in `color_dict`.
    """

    color_dict_out = {
        (key, True): color for key, color in color_dict.items()
    }

    for key, color in color_dict.items():
        color_dict_out[(key, False)] = pavlib.fig.lighten_color(color, lighten_amt)

    return color_dict_out

def color_to_ucsc_string(color):
    """
    Convert a matplotlib color to a UCSC color string.

    :param color: Tuple of RGB values (0.0-1.0) or a dict with tuples of RGB values.

    :return: Color string (if `color` is a tuple) or color dictionary of strings (if `color` is a dictionary of
        tuple values).
    """

    if isinstance(color, dict):
        return {
            key: ','.join((str(int(color_val * 255)) for color_val in color_val_tup))
                for key, color_val_tup in color.items()
        }

    return ','.join((str(int(color_val * 255)) for color_val in color))

def _track_get_input_bed(wildcards):
    """
    Get one or more input files for tracks. If "svtype" is "all", collect all relevant input files.

    :param wildcards: Wildcards.

    :return: List of input file(s).
    """

    # Get input file variant type
    if wildcards.vartype in {'sv', 'indel', 'svindel'}:


        if wildcards.svtype in {'ins', 'del'}:
            input_vartype = 'svindel'
            input_svtype = [wildcards.vartype]
        elif wildcards.svtype == 'insdel':
            input_vartype = 'svindel'
            input_svtype = ['ins', 'del']
        elif wildcards.svtype == 'inv':
            input_vartype = 'sv'
            input_svtype = ['inv']
        else:
            raise RuntimeError(f'Unknown svtype {wildcards.svtype} for variant type {wildcards.vartype} (expected "ins", "del", or "insdel" - "inv" allowed for vartype "sv")')

        if 'inv' in input_svtype and input_vartype not in {'sv', 'svindel'}:
            raise RuntimeError(f'Bad svtype {wildcards.svtype} for variant type {wildcards.vartype}: vartype must include SVs to output inversions')

    elif wildcards.vartype == 'snv':
        if wildcards.svtype != 'snv':
            raise RuntimeError(f'Unknown svtype {wildcards.svtype} for variant type {wildcards.vartype} (expected "snv")')

        input_vartype = 'snv'
        input_svtype = ['snv']

    else:
        raise RuntimeError(f'Unrecognized variant type: {wildcards.vartype}')

    # Get filter list
    if wildcards.filter == 'pass':
        input_filter = ['pass']
    elif wildcards.filter == 'fail':
        input_filter = ['fail']
    elif wildcards.filter == 'all':
        input_filter = ['pass', 'fail']
    else:
        raise RuntimeError(f'Unknown input filter (wildcards.filter): "{wildcards.filter}"')

    # Return list of input files
    return [
        VAR_BED_PREMERGE_PATTERN.format(vartype=input_vartype, svtype=svtype, filter=filter)
            for svtype in input_svtype for filter in input_filter
    ]


#
# All
#

# All tracks
rule tracks_all:
    input:
        bb_call=lambda wildcards: pavlib.pipeline.expand_pattern(
            'tracks/{asm_name}/variant/pre_merge/pass/{varsvtype}_{hap}.bb', ASM_TABLE, config,
            varsvtype=['sv_insdel', 'sv_inv', 'indel_insdel', 'snv_snv']
        ),
        bb_align=lambda wildcards: pavlib.pipeline.expand_pattern(
            'tracks/{asm_name}/align/align_qry_trim-{trim}.bb', ASM_TABLE, config,
            trim=('none', 'qry', 'qryref')
        ),
        bb_invflag=lambda wildcards: pavlib.pipeline.expand_pattern(
            'tracks/{asm_name}/inv_flag/inv_flag.bb', ASM_TABLE, config
        )

rule tracks_bb:
    input:
        bed='temp/{asm_name}/tracks/{subdir}/{filename}.bed',
        asfile='temp/{asm_name}/tracks/{subdir}/{filename}.as',
        fai=REF_FAI
    output:
        bb='tracks/{asm_name}/{subdir}/{filename}.bb'
    shell:
        """bedToBigBed -tab -as={input.asfile} -type=bed9+ {input.bed} {input.fai} {output.bb}"""

#
# Variant calls
#

rule tracks_hap_call_all:
    input:
        bb=lambda wildcards: pavlib.pipeline.expand_pattern(
            'tracks/{asm_name}/variant/pre_merge/pass/{varsvtype}_{hap}.bb', ASM_TABLE, config,
            varsvtype=['sv_insdel', 'sv_inv', 'indel_insdel', 'snv_snv']
        )

# # BigBed for one variant set.
# rule tracks_hap_call_bb:
#     input:
#         bed='temp/{asm_name}/tracks/bed_pre_merge/{filter}/{vartype}_{svtype}_{hap}.bed',
#         asfile='temp/{asm_name}/tracks/bed_pre_merge/{filter}/{vartype}_{svtype}_{hap}.as',
#         fai=REF_FAI
#     output:
#         bb='tracks/{asm_name}/variant/pre_merge/{filter}/{vartype}_{svtype}_{hap}.bb'
#     shell:
#         """bedToBigBed -tab -as={input.asfile} -type=bed9+ {input.bed} {input.fai} {output.bb}"""

# Tracks for one variant set.
rule tracks_hap_call:
    input:
        bed=_track_get_input_bed,
        fai=REF_FAI
    output:
        bed=temp('temp/{asm_name}/tracks/bed_pre_merge/{filter}/{vartype}_{svtype}_{hap}.bed'),
        asfile=temp('temp/{asm_name}/tracks/bed_pre_merge/{filter}/{vartype}_{svtype}_{hap}.as')
    wildcard_constraints:
        filter='pass|fail|all'
    run:

        if wildcards.filter != 'pass':
            raise NotImplementedError(f'Tracks containing non-PASS variants is not yet supported: {wildcards.filter}')

        field_table_file_name = os.path.join(PIPELINE_DIR, 'files/tracks/variant_track_fields.tsv')

        # Read variants
        # noinspection PyTypeChecker
        df = pd.concat(
            [pd.read_csv(file_name, sep='\t', dtype={'#CHROM': str}, low_memory=False) for file_name in input.bed],
            axis=0
        ).reset_index(drop=True)

        # noinspection PyUnresolvedReferences
        if wildcards.vartype == 'sv':
            df = df.loc[df['SVLEN'] >= 50].copy()
        elif wildcards.vartype == 'indel':
            df = df.loc[df['SVLEN'] < 50].copy()

        df.sort_values(['#CHROM', 'POS', 'END'], inplace=True)

        # Select table columns
        for col in ('QRY_ID', 'QRY_POS', 'QRY_END', 'QRY_MAPPED', 'QRY_STRAND', 'SEQ'):
            if col in df.columns:
                del(df[col])

        # Read FAI and table columns
        df_fai = svpoplib.ref.get_df_fai(input.fai)

        # Filter columns that have track annotations
        field_set = set(
            pd.read_csv(
                field_table_file_name,
                sep='\t', header=0
            )['FIELD']
        )

        df = df.loc[:, [col for col in df.columns if col in field_set]]

        # Make BigBed
        track_name = 'VariantTable'
        track_description = '{asm_name} - {vartype}-{svtype} - {hap}'.format(**wildcards)

        svpoplib.tracks.variant.make_bb_track(df, df_fai, output.bed, output.asfile, track_name, track_description, field_table_file_name)


#
# Alignments
#

# Generate all alignment tracks
rule tracks_align_all:
    input:
        bb=lambda wildcards: pavlib.pipeline.expand_pattern(
            'tracks/{asm_name}/align/align_qry_trim-{trim}.bb', ASM_TABLE, config,
            trim=('none', 'qry', 'qryref')
        )

# # Alignment track BED to BigBed.
# rule tracks_align_bb:
#     input:
#         bed='temp/{asm_name}/tracks/align/align_qry_trim-{trim}.bed',
#         asfile='temp/{asm_name}/tracks/align/align_qry_trim-{trim}.as'
#     output:
#         bb='tracks/{asm_name}/align/align_qry_trim-{trim}.bb'
#     shell:
#         """bedToBigBed -tab -as={input.asfile} -type=bed9+ {input.bed} {REF_FAI} {output.bb}"""

# Alignment tracks.
rule tracks_align:
    # noinspection PyUnresolvedReferences
    input:
        bed=lambda wildcards: [
            f'results/{wildcards.asm_name}/align/trim-{wildcards.trim}/align_qry_{hap}.bed.gz'
                for hap in pavlib.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE)
        ]
    output:
        bed=temp('temp/{asm_name}/tracks/align/align_qry_trim-{trim}.bed'),
        asfile=temp('temp/{asm_name}/tracks/align/align_qry_trim-{trim}.as')
    run:

        # Get track description
        # noinspection PyUnresolvedReferences
        if wildcards.trim == 'none':
            track_desc_short = f'PavAlignNone'
            track_description = f'PAV Align (Trim NONE)'

        elif wildcards.trim == 'qry':
            track_desc_short = f'PavAlignQry'
            track_description = f'PAV Align (Trim QRY)'

        elif wildcards.trim == 'qryref':
            track_desc_short = f'PavAlignQryref'
            track_description = f'PAV Align (Trim QRY/REF)'

        else:
            # noinspection PyUnresolvedReferences
            raise RuntimeError('Unknown trim wildcard: '.format(wildcards.trim))

        # Read field table
        df_as = pd.read_csv(
            os.path.join(PIPELINE_DIR, 'files/tracks/alignment_track_fields.tsv'),
            sep='\t'
        ).set_index('FIELD')

        # Read alignments
        # noinspection PyTypeChecker
        df = pd.concat(
            [pd.read_csv(file_name, sep='\t', dtype={'#CHROM': str, 'QRY_ID': str}) for file_name in input.bed],
            axis=0
        )

        del df['CIGAR']

        # Set Filter
        if 'FILTER' not in df.columns:
            df['FILTER'] = 'PASS'

        df['FILTER'] = df['FILTER'].fillna('PASS')

        # Sort
        df.sort_values(['#CHROM', 'POS', 'END', 'QRY_ID'], inplace=True)

        df.reset_index(drop=True, inplace=True)

        # Add BED fields
        df['POS_THICK'] = df['POS']
        df['END_THICK'] = df['END']
        df['ID'] = df.apply(lambda row: '{QRY_ID} - {INDEX} ({HAP}-{QRY_ORDER})'.format(**row), axis=1)

        if 'MATCH_PROP' in df.columns:
            df['TRK_SCORE'] = (
                np.clip(
                    df['MATCH_PROP'].fillna(0.0) * 1000, 0.0, 1000.0
                )
            ).astype(int)
        else:
            df['TRK_SCORE'] = 1000


        df['STRAND'] = df['IS_REV'].apply(lambda val: '-' if val else '+')

        # Set Color
        # noinspection PyUnresolvedReferences
        hap_list = pavlib.pipeline.get_hap_list(wildcards.asm_name, ASM_TABLE)
        colormap_index = np.linspace(0, 0.9999, len(hap_list))
        colormap = mpl.colormaps[ALIGN_COLORMAP]

        # hap_color = {  # Color to RGB string (e.g. "(0.267004, 0.004874, 0.329415, 1.0)" from colormap to "68,1,84")
        #     hap_list[i]: ','.join([str(int(col * 255)) for col in mpl.colors.to_rgb(colormap(colormap_index[i]))])
        #         for i in range(len(hap_list))
        # }

        hap_color_pass = {  # Color to RGB string (e.g. "(0.267004, 0.004874, 0.329415, 1.0)" from colormap to "68,1,84")
            (hap_list[i], True): mpl.colors.to_rgb(colormap(colormap_index[i])) for i in range(len(hap_list))
        }

        hap_color_fail = {  # Lighter color for filtered alignments
            (hap_name, False): pavlib.fig.util.lighten_color(color, 0.25) for (hap_name, is_pass), color in hap_color_pass.items()
        }

        hap_color = pd.Series(
            hap_color_pass | hap_color_fail,
            name='COL'
        )

        hap_color.index = hap_color.index.set_names(('HAP', 'PASS'))

        hap_color = hap_color.apply(lambda val: ','.join([str(int(col * 255)) for col in val]))

        df['COL'] = df.apply(lambda row:
            hap_color[(row['HAP'], row['FILTER'] == 'PASS')],
            axis=1
        )

        # Sort columns
        head_cols = ['#CHROM', 'POS', 'END', 'ID', 'TRK_SCORE', 'STRAND', 'POS_THICK', 'END_THICK', 'COL']
        tail_cols = [col for col in df.columns if col not in head_cols]

        df = df[head_cols + tail_cols]

        # Check AS fields
        missing_fields = [col for col in df.columns if col not in df_as.index]

        if missing_fields:
            raise RuntimeError('Missing {} fields in AS definition: {}{}'.format(
                len(missing_fields), ', '.join(missing_fields[:3]), '...' if len(missing_fields) else ''
            ))

        # Write AS file
        with open(output.asfile, 'w') as out_file:

            # Heading
            out_file.write('table Align{}\n"{}"\n(\n'.format(track_desc_short, track_description))

            # Column definitions
            for col in df.columns:
                out_file.write('{TYPE} {NAME}; "{DESC}"\n'.format(**df_as.loc[col]))

            # Closing
            out_file.write(')\n')

        # Write BED
        df.to_csv(output.bed, sep='\t', index=False)

#
# Inversion flagged loci
#

rule tracks_invflag_all:
    input:
        bb=lambda wildcards: pavlib.pipeline.expand_pattern(
            'tracks/{asm_name}/inv_flag/inv_flag_{hap}.bb', ASM_TABLE, config
        )

# BigBed for one variant set.
rule tracks_invflag_bb:
    input:
        bed='temp/{asm_name}/tracks/inv_flag/inv_flag_{hap}.bed',
        asfile='temp/{asm_name}/tracks/inv_flag/inv_flag_{hap}.as',
        fai=REF_FAI
    output:
        bb='tracks/{asm_name}/inv_flag/inv_flag_{hap}.bb'
    shell:
        """bedToBigBed -tab -as={input.asfile} -type=bed9+ {input.bed} {input.fai} {output.bb}"""

# Tracks for one variant set.
rule tracks_invflag_bed:
    # noinspection PyUnresolvedReferences
    input:
        bed=lambda wildcards: 'results/{asm_name}/inv_caller/flagged_regions_{hap}_parts-{part_count}.bed.gz'.format(
            part_count=pavlib.pavconfig.ConfigParams(wildcards.asm_name, config, ASM_TABLE, verbose=False).inv_sig_part_count, **wildcards
        ),
        fai=REF_FAI
    output:
        bed=temp('temp/{asm_name}/tracks/inv_flag/inv_flag_{hap}.bed'),
        asfile=temp('temp/{asm_name}/tracks/inv_flag/inv_flag_{hap}.as')
    run:

        field_table_file_name = os.path.join(PIPELINE_DIR, 'files/tracks/inv_flag_fields.tsv')

        color = {
            True: '0,0,0',
            False: '120,120,120'
        }

        # Read variants
        # noinspection PyTypeChecker
        df = pd.read_csv(input.bed, sep='\t')

        df.sort_values(['#CHROM', 'POS', 'END'], inplace=True)

        # Set color
        df['COL'] = df['PARTITION'].apply(lambda val: color[val >= 0])

        # Read FAI and table columns
        df_fai = svpoplib.ref.get_df_fai(input.fai)

        # Filter columns that have track annotations
        field_set = set(
            pd.read_csv(
                field_table_file_name,
                sep='\t', header=0
            )['FIELD']
        )

        df = df.loc[:, [col for col in df.columns if col in field_set]]

        # Make BigBed
        track_name = 'InvFlagTable'
        track_description = '{asm_name} - {hap}'.format(**wildcards)

        svpoplib.tracks.variant.make_bb_track(df, df_fai, output.bed, output.asfile, track_name, track_description, field_table_file_name)

#
# LG-SV
#

rule tracks_lgsv_all:
    input:
        bb_cpx=lambda wildcards: pavlib.pipeline.expand_pattern(
            'tracks/{asm_name}/lgsv/lgsv_cpx_{hap}.bb', ASM_TABLE, config
        ),
        bb_insdelinv=lambda wildcards: pavlib.pipeline.expand_pattern(
            'tracks/{asm_name}/lgsv/lgsv_insdelinv_{hap}.bb', ASM_TABLE, config
        )

rule tracks_lgsv_cpx_bed:
    input:
        bed_cpx='results/{asm_name}/lgsv/sv_cpx_{hap}.bed.gz',
        bed_seg='results/{asm_name}/lgsv/segment_{hap}.bed.gz',
        bed_cpx_ref='results/{asm_name}/lgsv/reftrace_cpx_{hap}.bed.gz',
        fai=REF_FAI
    output:
        bed=temp('temp/{asm_name}/tracks/lgsv/lgsv_cpx_{hap}.bed'),
        asfile=temp('temp/{asm_name}/tracks/lgsv/lgsv_cpx_{hap}.as')
    run:

        field_table_file_name = os.path.join(PIPELINE_DIR, 'files/tracks/lgsv_track_fields.tsv')

        # Read variants
        df_cpx = pd.read_csv(input.bed_cpx, sep='\t', dtype={'#CHROM': str}, low_memory=False)
        df_cpx.sort_values(['#CHROM', 'POS', 'END'], inplace=True)

        if 'FILTER' not in df_cpx.columns:
            df_cpx['FILTER'] = 'PASS'
        else:
            df_cpx['FILTER'] = df_cpx['FILTER'].fillna('PASS').astype(str)

        # # Set colors
        # if df_cpx.shape[0] > 0:
        #     color_dict = get_colors_pass_fail(CPX_COLOR_DICT)
        #     df_cpx['COL'] = df_cpx.apply(lambda row: color_dict[(row['TYPE'], row['FILTER'] == 'PASS')], axis=1)
        # else:
        #     df_cpx['COL'] = np.nan

        # Read structure tables
        df_rt_all = pd.read_csv(
            input.bed_cpx_ref, sep='\t', low_memory=False,
            usecols=['#CHROM', 'POS', 'END', 'ID', 'TYPE', 'DEPTH', 'INDEX', 'FWD_COUNT', 'REV_COUNT']
        )[['#CHROM', 'POS', 'END', 'ID', 'TYPE', 'DEPTH', 'INDEX', 'FWD_COUNT', 'REV_COUNT']]

        df_seg_all = pd.read_csv(input.bed_seg, sep='\t', low_memory=False)

        # Read FAI and table columns
        df_fai = svpoplib.ref.get_df_fai(input.fai)

        # Build track table (mix of record types for each complex variant - DEL, DUP, etc)
        df_track_list = list()

        for index, row in df_cpx.iterrows():

            df_rt = df_rt_all.loc[df_rt_all['ID'] == row['ID']].copy()
            df_seg = df_seg_all.loc[df_seg_all['ID'] == row['ID']]

            if df_seg.shape[0] == 0:
                raise RuntimeError(f'Variant {row["ID"]} not found in the CPX segment table')

            # Add unmapped segments to reference trace
            pos = None
            trace_list = list()

            for i in range(df_seg.shape[0]):
                if not df_seg.iloc[i]['IS_ALIGNED'] and not df_seg.iloc[i]['IS_ANCHOR']:
                    if pos is None:
                        raise RuntimeError(f'Segment at position {i} is not aligned and is not preceded by aligned segments')

                    row_seg = df_seg.iloc[i]

                    trace_list.append(
                        pd.Series(
                            [row['#CHROM'], pos, pos + row_seg['LEN_QRY'], row['ID'], 'UNMAPPED', 0, '.', 0, 0],
                            index=['#CHROM', 'POS', 'END', 'ID', 'TYPE', 'DEPTH', 'INDEX', 'FWD_COUNT', 'REV_COUNT']
                        )
                    )

                else:
                    if df_seg.iloc[i]['#CHROM'] == row['#CHROM']:
                        pos = df_seg.iloc[i]['END']

            if len(trace_list) > 0:
                df_rt = pd.concat([df_rt, pd.concat(trace_list, axis=1).T], axis=0)

            for col in ('FILTER', 'QRY_REGION', 'QRY_STRAND', 'SEG_N', 'STRUCT_REF', 'STRUCT_QRY', 'VAR_SCORE', 'ANCHOR_SCORE_MIN', 'ANCHOR_SCORE_MAX'):
                df_rt[col] = row[col] if col in row else np.nan

            df_rt.reset_index(drop=True, inplace=True)

            if df_rt.shape[0] > 0:
                df_rt['ID'] = df_rt.apply(lambda row_rt: row_rt['ID'] + f' ({row_rt.name + 1} / {df_rt.shape[0]})', axis=1)

                df_track_list.append(df_rt)

        if len(df_track_list) > 0:
            df = pd.concat(df_track_list, axis=0).reset_index(drop=True)
        else:
            df = pd.DataFrame([], columns=['#CHROM', 'POS', 'END', 'ID', 'TYPE', 'DEPTH', 'INDEX', 'FWD_COUNT', 'REV_COUNT'])

        # Truncate records that extend off the ends of chromosomes
        if df.shape[0] > 0:
            df_err = df.loc[df.apply(lambda row: row['POS'] < 0, axis=1)]

            if np.any(df_err):
                raise RuntimeError(f'Found {df_err.shape[0]} records with negative POS')

            df['END'] = df.apply(lambda row: np.min([row['END'], df_fai.loc[row['#CHROM']]]), axis=1)

        # Set colors
        if df.shape[0] > 0:
            color_dict = color_to_ucsc_string(get_colors_pass_fail(CPX_COLOR_DICT))
            df['COL'] = df.apply(lambda row: color_dict[(row['TYPE'], row['FILTER'] == 'PASS')], axis=1)
        else:
            df['COL'] = np.nan

        # Add bed track fields
        df['SCORE'] = 1000
        df['STRAND'] = '.'
        df['POS_THICK'] = df['POS']
        df['END_THICK'] = df['END']

        # Arrange columns
        head_cols = ['#CHROM', 'POS', 'END', 'ID', 'SCORE', 'STRAND', 'POS_THICK', 'END_THICK', 'COL', 'FILTER']
        tail_cols = [col for col in df.columns if col not in head_cols]

        df = df.loc[:, head_cols + tail_cols]

        df.sort_values(['#CHROM', 'POS', 'ID'], inplace=True)

        df['INDEX'] = df['INDEX'].fillna('.')


        ### Define AS columns (AutoSQL, needed to make a BigBed) ###

        # noinspection PyTypeChecker
        df_as = pd.read_csv(
            field_table_file_name,
            sep='\t', header=0,
            dtype={'DEFAULT': object},
            na_values=[''], keep_default_na=False
        )

        df_as.set_index('FIELD', inplace=True, drop=False)

        missing_list = [col for col in tail_cols if col not in set(df_as['FIELD'])]

        if missing_list:
            raise RuntimeError('Missing AS definitions for columns: {}'.format(', '.join(missing_list)))

        # Reformat columns
        for col in df.columns:
            if col == '#CHROM':
                if np.any(pd.isnull(df[col])):
                    raise RuntimeError(f'Error formatting {col}: Found null values in this column (not allowed)')

                continue

            if 'DEFAULT' in df_as.columns and not pd.isnull(df_as.loc[col, 'DEFAULT']):
                default_val = df_as.loc[col, 'DEFAULT']
            else:
                default_val = '.'

            format_type = svpoplib.tracks.variant.TYPE_DICT.get(df_as.loc[col, 'TYPE'], str)

            try:
                df[col] = svpoplib.tracks.variant.format_column(df[col], svpoplib.tracks.variant.TYPE_DICT.get(df_as.loc[col, 'TYPE'], str), default_val=default_val)
            except Exception as ex:
                raise RuntimeError('Error formatting {} as {}: {}'.format(col, df_as.loc[col, 'TYPE'], ex))

        # Write
        track_name = 'LGSVVariantTableCPX'
        track_description = '{asm_name} - LG-SV CPX - {hap}'.format(**wildcards)

        with open(output.asfile, 'w') as out_file:
            # Heading
            out_file.write('table {}\n"{}"\n(\n'.format(track_name, track_description))

            # Column definitions
            for col in head_cols + tail_cols:
                out_file.write('{TYPE} {NAME}; "{DESC}"\n'.format(**df_as.loc[col]))

            # Closing
            out_file.write(')\n')

        df.to_csv(output.bed, sep='\t', na_rep='.', index=False)


rule tracks_lgsv_insdelinv_bed:
    input:
        bed_ins='results/{asm_name}/lgsv/svindel_ins_{hap}.bed.gz',
        bed_del='results/{asm_name}/lgsv/svindel_del_{hap}.bed.gz',
        bed_inv='results/{asm_name}/lgsv/sv_inv_{hap}.bed.gz',
        fai=REF_FAI
    output:
        bed=temp('temp/{asm_name}/tracks/lgsv/lgsv_insdelinv_{hap}.bed'),
        asfile=temp('temp/{asm_name}/tracks/lgsv/lgsv_insdelinv_{hap}.as')
    run:

        field_table_file_name = os.path.join(PIPELINE_DIR, 'files/tracks/lgsv_track_fields.tsv')

        # Read variants
        df = pd.concat(
            [
                pd.read_csv(filename, sep='\t', dtype={'#CHROM': str}, low_memory=False)
                    for filename in [input.bed_ins, input.bed_del, input.bed_inv]
            ]
        ).sort_values(['#CHROM', 'POS', 'END'])

        if 'DUP_REGION' in df.columns:
            # TEMP: Eliminate once DUP_REGION has been renamed by the LGSV caller
            df.columns = [{'DUP_REGION': 'TEMPL_REGION'}.get(col, col) for col in df.columns]

        elif 'TEMPL_REGION' not in df.columns:
            df['TEMPL_REGION'] = np.nan

        # Read FAI and table columns
        df_fai = svpoplib.ref.get_df_fai(input.fai)

        # Filter columns that have track annotations
        field_set = set(
            pd.read_csv(
                field_table_file_name,
                sep='\t', header=0
            )['FIELD']
        )

        col_order = [col for col in df.columns if col in field_set]

        df = df.loc[:, col_order]

        # Add templated loci
        df_templ_list = list()

        for index, row in df.loc[~ pd.isnull(df['TEMPL_REGION'])].iterrows():

            templ_region_list = [
                pavlib.seq.region_from_string(val_strip) for val in row['TEMPL_REGION'].split(',') if (val_strip := val.strip())
            ]

            n_region = len(templ_region_list)
            n = 0

            for templ_region in templ_region_list:
                n += 1
                templ_row = row.copy()

                templ_row['ID'] = row['ID'].strip() + f' templ ({n} / {n_region})'
                templ_row['SVTYPE'] = 'DUP'

                df_templ_list.append(templ_row)

        if len(df_templ_list) > 0:
            df = pd.concat([df, pd.DataFrame(df_templ_list)]).sort_values(['#CHROM', 'POS', 'END'])

        # Make BigBed
        track_name = 'LGSVVariantTableSV'
        track_description = '{asm_name} - LG-SV INS/DEL/INV - {hap}'.format(**wildcards)

        svpoplib.tracks.variant.make_bb_track(df, df_fai, output.bed, output.asfile, track_name, track_description, field_table_file_name)
