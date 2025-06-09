"""
Variant caller functions.
"""


import collections
import intervaltree
import numpy as np
import pandas as pd
import polars as pl
import re

from collections.abc import Callable

import svpoplib

from . import align
from . import util
from . import pavconfig


# Explanations for filter codes
FILTER_REASON = {
    align.records.FILTER_LCALIGN: 'Variant inside a low-confidence alignment record',
    align.records.FILTER_ALIGN: 'Variant inside an alignment record that had a filtered flag (matches 0x700 in alignment flags) or did not meet a minimum MAPQ (not set by default)',
    'PASS': 'Variant passed filters',
    'QRY_FILTER': 'Query filter region (regions provided to PAV at runtime)',
    'PURGE': 'Inside larger variant',  # Previously "COMPOUND"
    'INNER': 'Part of a larger variant call (i.e. a variant inside of a larger duplication)',
    'SVLEN': 'Variant size out of set bounds (sizes set in the PAV config file)',
    'TRIMREF': 'Alignment trimming in reference coordinates removed variant',
    'TRIMQRY': 'Alignment trimming in query coordinates removed variant'
}

# Types for known Pandas columns
PD_COL_TYPES = {
    '#CHROM': np.str_,
    'POS': np.uint32,
    'END': np.uint32,
    'ID': np.str_,
    'SVTYPE': np.str_,
    'SVLEN': np.uint32,
    'FILTER': np.str_,
    'HAP': np.str_,
    'QRY_ID': np.str_,
    'QRY_POS': np.uint32,
    'QRY_END': np.uint32,
    'QRY_STRAND': np.str_,
    'CI': np.str_,
    'ALIGN_INDEX': np.str_,
    'LEFT_SHIFT': np.uint32,
    'HOM_REF': np.str_,
    'HOM_TIG': np.str_,
    'CALL_SOURCE': np.str_,
    'SEQ': np.str_,
    'REF': np.str_,
    'ALT': np.str_,
    'RESOLVED_TEMPL': np.str_,
    'SVSUBTYPE': np.str_,
    'TEMPL_REGION': np.str_,
    'VAR_SCORE': np.float32,
    'ANCHOR_SCORE_MIN': np.float32,
    'ANCHOR_SCORE_MAX': np.float32,
    'INTERVAL': np.str_,
    'REGION_REF_OUTER': np.str_,
    'REGION_QRY_OUTER': np.str_,
    'RGN_REF_OUTER': np.str_,
    'RGN_QRY_OUTER': np.str_,
    'ALIGN_SIZE_GAP': np.uint32,
    'SEG_N': np.uint32,
    'STRUCT_REF': np.str_,
    'STRUCT_QRY': np.str_,
    'FLAG_TYPE': np.str_,
    'VAR_SOURCE': np.str_
}


def version_variant_bed_id(df, re_version=False):
    """
    Version IDs in a variant call BED table (pd.Series). Table should have "ID", "FILTER", "QRY_REGION", "QRY_POS", and
    "QRY_END" fields. If ID is missing, it will be generated from the table. Variants are prioritized by FILTER=PASS
    with "dot" versions (e.g. ID.1, ID.2) being preferentially added to the variant call that do not have FILTER=PASS.

    :param df: Variant call DataFrame.
    :param re_version: If True, eliminate existing versions from IDs before re-versioning.

    :return: A Series object with the new IDs.
    """

    # Determine which columns are available for sorting.
    get_cols = [col for col in ['ID', 'FILTER', 'QRY_ID', 'QRY_POS', 'QRY_END'] if col in df.columns]

    if 'ID' not in get_cols:
        id_col = svpoplib.variant.get_variant_id(df, apply_version=False)
    else:
        id_col = None

    # Subset columns
    if len(get_cols) > 0:
        # Make a copy (order is changed)
        df_re = df[get_cols].copy().reset_index()

        if id_col is not None:
            df_re['ID'] = list(id_col)
            get_cols = ['ID'] + get_cols
    else:
        # Create with a single ID column
        df_re = pd.DataFrame(id_col)
        df_re.columns = ['ID']

        get_cols = ['ID']

    if 'FILTER' not in get_cols:
        df_re['FILTER'] = 'PASS'

    # Remove versions (.1, .2, etc) if present
    if re_version:
        df_re['ID'] = df_re['ID'].apply(lambda val: val.rsplit('.', 1)[0])

    # Sort by ID, FILTER, then query region
    df_re['FILTER'].fillna('')
    df_re['FILTER'] = df_re['FILTER'].apply(lambda val: ('a' if val == 'PASS' else 'b') + val)  # Force PASS to sort

    df_re.sort_values(['ID', 'FILTER', 'QRY_ID', 'QRY_POS'], inplace=True)

    # Re-ID, first PASS, then non-PASS
    df_re.loc[df_re['FILTER'] == 'aPASS', 'ID'] = svpoplib.variant.version_id(
        df_re.loc[df_re['FILTER'] == 'aPASS', 'ID']
    )

    df_re.loc[df_re['FILTER'] != 'aPASS', 'ID'] = svpoplib.variant.version_id(
        df_re.loc[df_re['FILTER'] != 'aPASS', 'ID'],
        existing_id_set=set(df_re.loc[df_re['FILTER'] == 'aPASS', 'ID'])
    )

    # Return IDs
    df_re.sort_index(inplace=True)
    df_re.index = df.index

    return df_re['ID']


def get_gt(row, hap, map_tree):
    """
    Get variant genotype based on haplotype and mappability.

    :param row: Variant call row (post h1/h2 merge).
    :param hap: Haplotype being called.
    :param map_tree: Tree of mappable regions.

    :return: '1' if the variant call is in this haplotype, '0' if it was not called but is in mappable regions,
        and '.' if it is not in a mappable region.
    """

    if hap in row['HAP'].split(';'):
        return '1'

    for interval in map_tree[row['#CHROM']].overlap(row['POS'], row['END']):

        if (interval.begin <= row['POS']) and (interval.end >= row['END']):
            return '0'

    return '.'


def val_per_hap(df, df_dict, col_name, delim=';'):
    """
    Construct a field from a merged variant DataFrame (`df`) by pulling values from each pre-merged haplotype. Matches
    the merged variant IDs from df to the correct original ID in each haplotype and returns a comma-separated string
    of values. `df_h1` and `df_h2` must be indexed with the original variant ID from the corresponding haplotype.
    Function returns a Pandas Series keyed by the `df` index.

    :param df: Merged DataFrame of variants.
    :param df_dict: Dictionary of pre-merged DataFrames keyed by haplotype name.
    :param col_name: Get this column name from `df_h1` and `df_h2`.
    :param delim: Separate values by this delimiter.

    :return: A Pandas series keyed by indices from `df` with comma-separated values extracted from each haplotype
        DataFrame.
    """

    # Generate a Series keyed by variant ID with tuples of (hap, id) for the haplotype and variant ID in that haplotype.
    # Then, apply a function to join the target values from each dataframe in the correct order.
    return df.apply(lambda row:
         tuple(zip(
             row['HAP'].split(';'), row['HAP_VARIANTS'].split(';')
         )),
         axis=1
    ).apply(
        lambda val_list: delim.join(str(df_dict[val[0]].loc[val[1], col_name]) for val in val_list)
    )


# def filter_by_ref_tree(row, filter_tree, reason, match_tig=False):
#     """
#     Filter DataFrame by a dict (keyed by chromosome) of interval trees.
#
#     `filter_tree` is a dictionary with one key per chromosome and an IntervalTree object as the value for each
#     chromosome. IntervalTrees have a start, end, and data. Elements in the filter tree are usually constructed from
#     the genomic locations of variant calls. The data should be a tuple of two elements: 1) The contig where the variant
#     was called from, and 2) The ID of the variant. The first element is needed for `match_tig`, and the second is needed
#     to annotate why variants are being dropped if they are inside another.
#
#     :param row: Variant call row.
#     :param filter_tree: Dict of trees, one entry per chromosome.
#     :param reason: Filter reason. "FILTER" is set to `reason` if it is "PASS", otherwise, it is appended to the
#         filter column with a comma. (e.g. "REASON1,REASON2").
#     :param match_tig: Match QRY_REGION contig name to the value in the filter tree for intersected intervals (filter
#         tree values should be contig names for the interval). When set, only filters variants inside regions with a
#         matching contig. This causes PAV to treat each contig as a separate haplotype.
#
#     :return: Filtered DataFrame.
#     """
#
#     intersect_set = filter_tree[row['#CHROM']][row['POS']:row['END']]
#
#     if intersect_set and match_tig:
#         intersect_set = {record for record in intersect_set if record.data[0] != row['QRY_ID']}
#
#     if intersect_set:
#         row['FILTER'] = 'COMPOUND'
#         row['COMPOUND'] = ','.join(val.data[1] for val in intersect_set)
#
#         if row['FILTER'] == 'PASS':
#             return reason
#
#         return row['FILTER'] + f',{reason}'
#
#     return row['REASON']  # Leave unaltered


# def filter_by_tig_tree(row, filter_tree, reason):
#     """
#     Filter records from a callset DataFrame by matching "QRY_REGION" with regions in an IntervalTree.
#
#     :param row: DataFrame to filter. Must contain field "QRY_REGION" and "FILTER" initialized to "PASS" (may contain
#         other non-PASS values).
#     :param filter_tree: A `collections.defaultdict` of `intervaltree.IntervalTree` (indexed by contig name) of
#         no-call regions. Variants with a tig region intersecting these records will be removed (any intersect). If
#         `None`, then `df` is not filtered.
#     :param reason: Filter reason. "FILTER" is set to `reason` if it is "PASS", otherwise, it is appended to the
#         filter column with a comma. (e.g. "REASON1,REASON2").
#
#     :return: Filter column value.
#     """
#
#     match_obj = re.match(r'^([^:]+):(\d+)-(\d+)$', row['QRY_REGION'])
#
#     if match_obj is None:
#         raise RuntimeError('Unrecognized QRY_REGION format for record {}: {}'.format(row.name, row['QRY_REGION']))
#
#     if filter_tree[match_obj[1]][int(match_obj[2]) - 1:int(match_obj[3])]:
#         if row['FILTER'] == 'PASS':
#             return reason
#
#         return row['FILTER'] + f',{reason}'
#
#     return row['FILTER']  # Leave unaltered


def read_variant_table(filename_list, drop_dup_id=False, version_id=True):
    """
    Read a variant table and prepare for callset integration. Returns the table and a `collections.defaultdict(set)`
    object one for filters found in the callset. Callset integration uses these objects to track
    variant filters (filters) and IDs of variants that cover this variant (compound) (i.e. the ID of a large DEL a
    small variant appears in will appear in the small variant's COMPOUND column and "COMPOUND" will be added to
    filters).

    :param filename_list: List of filenames (list) or a single filename (str) to read.
    :param drop_dup_id: If `True`, drop duplicates by variant ID keeping the first variant in the table. If `False`,
        duplicate IDs are versioned (ID.1, ID.2).
    :param version_id: If `True`, version variants with duplicate IDs. Has no effect if `drop_dup_id` is `True`.

    :return: A tuple of [0] variant table, [1] filter dict, [2] compound dict.
    """

    if isinstance(filename_list, str):
        filename_list = [filename_list]

    df_list = [
        pd.read_csv(
            filename, sep='\t',
            low_memory=False, keep_default_na=False
        )
        for filename in filename_list
    ]

    if len(filename_list) > 1:
        df = pd.concat(df_list, axis=0)
    elif len(filename_list) == 1:
        df = df_list[0]
    else:
        raise RuntimeError('Filename list length is 0 or is not a list')

    df = df.sort_values(
        ['#CHROM', 'POS', 'END', 'ID']
    ).reset_index(drop=True)

    if drop_dup_id:
        df.drop_duplicates('ID', keep='first', inplace=True)

    if version_id and not drop_dup_id:
        df['ID'] = svpoplib.variant.version_id(df['ID'])

    df.set_index('ID', inplace=True, drop=False)
    df.index.name = 'INDEX'

    # Set filters
    filter_dict = collections.defaultdict(set)

    if 'FILTER' not in df.columns:
        df['FILTER'] = 'PASS'

    else:
        df['FILTER'] = df['FILTER'].fillna('PASS').astype(str).str.upper().str.strip()

        for var_id, filter_str in df['FILTER'].items():
            filter_set = set(filter_str.split(',')) - {'PASS', ''}

            if filter_set:
                filter_dict[var_id] = filter_set

    return df, filter_dict


# class DepthContainer:
#     """
#     Computes average alignment depth and coverage at SV sites from a coverage BED file.
#     """
#
#     def __init__(self, df_cov):
#         self.df_cov = df_cov
#
#         if df_cov is None or df_cov.shape[0] == 0:
#             raise RuntimeError('Coverage table is missing or empty')
#
#         # Make a table of indexes into df_cov
#         chrom = None
#
#         last_end = 0
#         first_index = 0
#
#         self.index_dict = dict()
#
#         for index in range(df_cov.shape[0]):
#             row = df_cov.iloc[index]
#
#             if row['#CHROM'] != chrom:
#
#                 if chrom is not None:
#                     self.index_dict[chrom] = (first_index, index)
#
#                 if row['#CHROM'] in self.index_dict:
#                     raise RuntimeError(f'Discontiguous chromosome order: Found {row["#CHROM"]} in multiple blocks')
#
#                 if row['POS'] != 0:
#                     raise RuntimeError(f'First record for chromosome {row["#CHROM"]} is not 0: {row["POS"]} (record location {index})')
#
#                 chrom = row['#CHROM']
#                 last_end = row['END']
#                 first_index = index
#
#             else:
#                 if row['POS'] != last_end:
#                     raise RuntimeError(f'Discontiguous or out of order record in {row["#CHROM"]} (record location {index}): POS={row["POS"]}, expected POS={last_end}')
#
#                 last_end = row['END']
#
#         assert chrom is not None, 'Missing chromosome at the end of the coverage table'
#
#         self.index_dict[chrom] = (first_index, df_cov.shape[0])
#
#         # Set state to the first chromosome
#         row = self.df_cov.iloc[0]
#
#         self.chrom = row['#CHROM']
#         self.index, self.last_index = self.index_dict[self.chrom]
#
#         self.pos = row['POS']
#         self.end = row['END']
#         self.depth = row['DEPTH']
#         self.qry_id = set(row['QRY_ID'].split(',')) if not pd.isnull(row['QRY_ID']) else set()
#
#     def get_depth(self, row):
#
#         # Switch chromosomes
#         if row['#CHROM'] != self.chrom:
#
#             if row['#CHROM'] not in self.index_dict:
#                 sv_id = row['ID'] if 'ID' in row.index else '<UNKNOWN>'
#                 raise RuntimeError(f'Variant "{sv_id}" (variant row index {row.name}) assigned to chromosome that is not in the depth table: {row["#CHROM"]}')
#
#             self.chrom = row['#CHROM']
#             self.index, self.last_index = self.index_dict[self.chrom]
#
#             self.pos = self.df_cov.iloc[self.index]['POS']
#             self.end = self.df_cov.iloc[self.index]['END']
#             self.depth = self.df_cov.iloc[self.index]['DEPTH']
#             self.qry_id = set(self.df_cov.iloc[self.index]['QRY_ID'].split(',')) if not pd.isnull(self.df_cov.iloc[self.index]['QRY_ID']) else set()
#
#         # Catch up to the coverage record where this record begins
#         #assert False, 'WORKING ON DEPTH CONTAINER'
#
#         is_end_ins = False
#
#         while row['POS'] >= self.end:
#             self.index += 1
#
#             if self.index >= self.last_index:
#
#                 # Rescue insertions added to the end of the chromosome. This can happen if the contig aligns up to
#                 # the end of the reference chromosome without clipping and with extra sequence added as an insertion
#                 # operation to the end of the alignment (CIGAR string). In this case, the "depth" is the number of
#                 # query records reaching the end of the reference chromosome that this insertion could have been
#                 # appended to.
#
#                 # self.pos, self.end, self.depth, and self.qry_id are already set
#
#                 if not (self.index == self.last_index and row['SVTYPE'] == 'INS' and row['END'] == row['POS'] + 1):
#                     # Variant is not in the bounds of this reference chromosome
#                     sv_id = row['ID'] if 'ID' in row.index else '<UNKNOWN>'
#                     raise RuntimeError(f'Ran out of depth records on "{self.chrom}" to the beginning of variant record {sv_id} (variant row index {row.name})')
#
#                 self.index -= 1
#                 is_end_ins = True
#                 break
#
#             self.pos = self.df_cov.iloc[self.index]['POS']
#             self.end = self.df_cov.iloc[self.index]['END']
#             self.depth = self.df_cov.iloc[self.index]['DEPTH']
#
#             self.qry_id = set(self.df_cov.iloc[self.index]['QRY_ID'].split(',')) if not pd.isnull(self.df_cov.iloc[self.index]['QRY_ID']) else set()
#
#         # If the variant fully contained within this coverage record, return stats from this region
#         if row['END'] < self.end or is_end_ins:
#             return self.depth, 1 if self.depth > 0 else 0, ','.join(sorted(self.qry_id))
#
#         # Get coverage from the variant position to the end of this coverage record
#         # sum_depth = self.depth * (self.end - row['POS'])
#         # sum_align = (1 if self.depth > 0 else 0) * (self.end - row['POS'])
#         # qry_id = set(self.qry_id.split(',')) if not pd.isnull(self.qry_id) else set()
#
#         sum_depth = 0
#         sum_align = 0
#         qry_id = set()
#
#         step_index = self.index
#
#         svlen = 0
#
#         last_end = self.df_cov.iloc[step_index]['POS']
#
#         # Get coverage from all coverage records fully contained within the variant record
#         while row['END'] > last_end:
#             if step_index >= self.last_index:
#                 sv_id = row['ID'] if 'ID' in row.index else '<UNKNOWN>'
#                 raise RuntimeError(f'Ran out of depth records on "{self.chrom}" to the end of variant record {sv_id} (variant row index {row.name})')
#
#             record_len = min(
#                 [row['END'], self.df_cov.iloc[step_index]['END']]
#             ) - max(
#                 [row['POS'], self.df_cov.iloc[step_index]['POS']]
#             )
#
#             last_end = self.df_cov.iloc[step_index]['END']
#
#             svlen += record_len
#
#             sum_depth += self.df_cov.iloc[step_index]['DEPTH'] * record_len
#
#             sum_align += record_len if self.df_cov.iloc[step_index]['DEPTH'] > 0 else 0
#
#             if not pd.isnull(self.df_cov.iloc[step_index]['QRY_ID']):
#                 qry_id |= set(self.df_cov.iloc[step_index]['QRY_ID'].split(',')) if not pd.isnull(self.df_cov.iloc[step_index]['QRY_ID']) else {}
#
#             step_index += 1
#
#             assert svlen == row['END'] - row['POS'], f'Record length for "{row["ID"] if "ID" in row.index else "<UNKNOWN>"}" after scanning a variant spanning multiple records does not match the expected variant length: Found {record_len}, expected {svlen}: Scanned depth table from index {self.index} to {step_index - 1} (inclusive)'
#
#         return (
#             sum_depth / svlen,
#             sum_align / svlen,
#             ','.join(sorted(qry_id)) if len(qry_id) > 0 else np.nan
#         )

def update_fields(
        df: pd.DataFrame,
        filter_dict: dict,
        purge_dict: dict,
        inner_dict: dict
) -> None:
    """
    Update FILTER, PURGE, and INNER fields in a variant call table.

    :param df: Variant call DataFrame.
    :param filter_dict: Dict keyed by variant IDs containing sets of FILTER strings.
    :param purge_dict: Dict keyed by variant IDs containing IDs of variants causing the record to be purged.

    :return: `df` with FILTER updated and PURGE added after the filter column.
    """

    # FILTER: If both TRIMREF and INNER, remove TRIMREF
    filter_dict = {
        key: (
            filter_set if len(filter_set & {'TRIMREF', 'INNER'}) == 2 else (
                filter_set - {'TRIMREF'}
            )
        )
            for key, filter_set in filter_dict.items()
    }

    df['FILTER'] = pd.Series(filter_dict).apply(
        lambda vals: ','.join(sorted(vals)) if vals else 'PASS'
    ).reindex(df.index, fill_value='PASS')

    col_purge = pd.Series(purge_dict).apply(
        lambda vals: ','.join(sorted(vals))
    ).reindex(df.index, fill_value='')

    col_inner = pd.Series(inner_dict).apply(
        lambda vals: ','.join(sorted(vals))
    ).reindex(df.index, fill_value='')

    if 'PURGE' in df.columns:
        df['PURGE'] = col_purge
    else:
        df.insert(
            [index for col, index in zip(df.columns, range(df.shape[1])) if col == 'FILTER'][0] + 1,
            'PURGE',
            col_purge
        )

    if 'INNER' in df.columns:
        df['INNER'] = col_inner
    else:
        df.insert(
            [index for col, index in zip(df.columns, range(df.shape[1])) if col == 'PURGE'][0] + 1,
            'INNER',
            col_inner
        )


def apply_purge_filter(
        df, purge_filter_tree, filter_dict, purge_dict, update=True, filter_str='PURGE'
):
    """
    Apply the purge variant filter. If a variant intersects one already seen in reference coordinates (i.e. a small
    variant inside a deletion), then "PURGE" is added to the filter (`filter_dict`) for that variant and the variant
    it intersected is added to `purge_dict` for that variant (key is variant ID, value is a set of variant IDs covering
    it).

    :param df: Variant DataFrame to filter.
    :param purge_filter_tree: Tree of existing variants to intersect.
    :param filter_dict: Filter dict matching variants in `df` keyed by variant IDs. Contains sets of filters for each
        variant. Adds `filter_str` for variants filtered by this method.
    :param purge_dict: Dictionary matching variants in `df` keyed by variant IDs. Contains sets of variant IDs
        "covering" this event (i.e. large variants already in the callset a variant intersected). The result of this
        can be used to reclaim smaller variants filtered by COMPOUND if all the variant IDs in it's `compound_dict`
        entry were removed from the callset. This dict is updated by this method.
    :param update: If `True`, update `compound_filter_tree` if a variant is not filtered and does not intersect another
        variant already in `compound_filter_tree`. Set to `False` if variants in `df` should not be intersected with
        future variants to test for compound filtering.
    :param filter_str: Adds this value to `filter_dict` for all variants filtered by this method. May not be `None` or
        an empty string.
    """

    # Check parameters
    if not isinstance(filter_str, str):
        raise RuntimeError('filter_str must be a string')

    filter_str = filter_str.strip()

    if not filter_str:
        raise RuntimeError('filter_str must not be empty')

    # Apply filter
    for index, row in df.sort_values(['SVLEN', 'POS'], ascending=(False, True)).iterrows():
        intersect_set = purge_filter_tree[row['#CHROM']][row['POS']:row['END']]
        filter_set = filter_dict.get(index, set())

        # Skip variants that are part of a larger variant
        if 'INNER' in filter_set:
            continue

        if len(intersect_set) > 0:
            # Filter variant
            filter_dict[index].add(filter_str)
            purge_dict[index] |= {val.data for val in intersect_set}
        else:
            # Add to filter regions if variant was not filtered
            if update and not filter_set:
                purge_filter_tree[row['#CHROM']][row['POS']:row['END']] = row['ID']

def apply_qry_filter_tree(df, qry_filter_tree, filter_dict):
    """
    Match the query coordinates from variant calls to a filter tree in query coordinates.

    :param df: Variant call dataframe.
    :param qry_filter_tree: Filter (dict keyed by query IDs with intervaltree objects covering filtered loci). If
        None, no filtering is applied (skips call).
    :param filter_dict: A dictionary (defaultdict) keyed by variant call IDs with sets of filter names. Filter
        "QRY_FILTER" is added to the set for all IDs intersecting a region in `qry_filter_tree`.
    """

    if qry_filter_tree is not None:

        filter_set = df.apply(
            lambda row: len(qry_filter_tree[row['QRY_ID']][row['QRY_POS']:row['QRY_END']]) > 0,
            axis=1
        )

        for index in filter_set[filter_set].index:
            filter_dict[index].add('QRY_FILTER')

def left_homology(pos_tig, seq_tig, seq_sv):
    """
    Determine the number of perfect-homology bp upstream of an SV/indel using the SV/indel sequence (seq_sv), a contig
    or reference sequence (seq_tig) and the position of the first base upstream of the SV/indel (pos_tig) in 0-based
    coordinates. Both the contig and SV/indel sequence must be in the same orientation (reverse-complement if needed).
    Generally, the SV/indel sequence is in reference orientation and the contig sequence is the reference or an
    aligned contig in reference orientation (reverse-complemented if needed to get to the + strand).

    This function traverses from `pos_tig` to upstream bases in `seq_tig` using bases from the end of `seq_sv` until
    a mismatch between `seq_sv` and `seq_tig` is found. Search will wrap through `seq_sv` if homology is longer than
    the SV/indel.

    WARNING: This function assumes upper-case for the sequences. Differing case will break the homology search. If any
    sequence is None, 0 is returned.

    :param pos_tig: Contig/reference position (0-based) in reference orientation (may have been reverse-complemented by an
        alignment) where the homology search begins.
    :param seq_tig: Contig sequence as an upper-case string and in reference orientation (may have been reverse-
        complemented by the alignment).
    :param seq_sv: SV/indel sequence as an upper-case string.

    :return: Number of perfect-homology bases between `seq_sv` and `seq_tig` immediately upstream of `pos_tig`. If any
        of the sequneces are None, 0 is returned.
    """

    if seq_sv is None or seq_tig is None:
        return 0

    svlen = len(seq_sv)

    hom_len = 0

    while hom_len <= pos_tig:  # Do not shift off the edge of a contig.
        seq_tig_base = seq_tig[pos_tig - hom_len]

        # Do not match ambiguous bases
        if seq_tig_base not in {'A', 'C', 'G', 'T'}:
            break

        # Match the SV sequence (dowstream SV sequence with upstream reference/contig)
        if seq_sv[-((hom_len + 1) % svlen)] != seq_tig_base:
            # Circular index through seq in reverse from last base to the first, then back to the first
            # if it wraps around. If the downstream end of the SV/indel matches the reference upstream of
            # the SV/indel, shift left. For tandem repeats where the SV was placed in the middle of a
            # repeat array, shift through multiple perfect copies (% oplen loops through seq).
            break

        hom_len += 1

    # Return shifted amount
    return hom_len


def right_homology(pos_tig, seq_tig, seq_sv):
    """
    Determine the number of perfect-homology bp downstream of an SV/indel using the SV/indel sequence (seq_sv), a contig
    or reference sequence (seq_tig) and the position of the first base downstream of the SV/indel (pos_tig) in 0-based
    coordinates. Both the contig and SV/indel sequence must be in the same orientation (reverse-complement if needed).
    Generally, the SV/indel sequence is in reference orientation and the contig sequence is the reference or an
    aligned contig in reference orientation (reverse-complemented if needed to get to the + strand).

    This function traverses from `pos_tig` to downstream bases in `seq_tig` using bases from the beginning of `seq_sv` until
    a mismatch between `seq_sv` and `seq_tig` is found. Search will wrap through `seq_sv` if homology is longer than
    the SV/indel.

    WARNING: This function assumes upper-case for the sequences. Differing case will break the homology search. If any
    sequence is None, 0 is returned.

    :param pos_tig: Contig/reference position (0-based) in reference orientation (may have been reverse-complemented by an
        alignment) where the homology search begins.
    :param seq_tig: Contig sequence as an upper-case string and in reference orientation (may have been reverse-
        complemented by the alignment).
    :param seq_sv: SV/indel sequence as an upper-case string.

    :return: Number of perfect-homology bases between `seq_sv` and `seq_tig` immediately downstream of `pos_tig`. If any
        of the sequences are None, 0 is returned.
    """

    if seq_sv is None or seq_tig is None:
        return 0

    svlen = len(seq_sv)
    tig_len = len(seq_tig)

    hom_len = 0
    pos_tig_limit = tig_len - pos_tig

    while hom_len < pos_tig_limit:  # Do not shift off the edge of a contig.
        seq_tig_base = seq_tig[pos_tig + hom_len]

        # Do not match ambiguous bases
        if seq_tig_base not in {'A', 'C', 'G', 'T'}:
            break

        # Match the SV sequence (dowstream SV sequence with upstream reference/contig)
        if seq_sv[hom_len % svlen] != seq_tig_base:
            # Circular index through seq in reverse from last base to the first, then back to the first
            # if it wraps around. If the downstream end of the SV/indel matches the reference upstream of
            # the SV/indel, shift left. For tandem repeats where the SV was placed in the middle of a
            # repeat array, shift through multiple perfect copies (% oplen loops through seq).
            break

        hom_len += 1

    # Return shifted amount
    return hom_len


def merge_haplotypes(
    bed_list,
    callable_list,
    hap_list,
    config_def,
    threads=1,
    subset_chrom=None
):
    """
    Merge haplotypes for one variant type.

    :param bed_list: List of input BED files.
    :param callable_list: List of callable reference loci in BED files.
    :param hap_list: List of haplotypes.
    :param config_def: Merge definition.
    :param threads: Number of threads for each merge.
    :param subset_chrom: Chromosome or set of chromosome names to merge, or `None` to merge all chromosomes in one step.

    :return: A dataframe of variant calls.
    """

    # Check input
    n_hap = len(hap_list)

    if len(bed_list) != n_hap:
        raise RuntimeError(f'Input variant BED list length ({len(bed_list)}) does not match the haplotype name list length: ({n_hap})')

    if len(callable_list) != n_hap:
        raise RuntimeError(f'Input callable BED list length ({len(callable_list)}) does not match the haplotype name list length: ({n_hap})')

    # Merge
    df = svpoplib.svmerge.merge_variants(
        bed_list=bed_list,
        sample_names=hap_list,
        strategy=config_def,
        threads=threads,
        subset_chrom=subset_chrom
    )

    df.set_index('ID', inplace=True, drop=False)
    df.index.name = 'INDEX'

    # Restructure columns
    for col in ('HAP', 'RGN_REF_DISC', 'RGN_QRY_DISC', 'FLAG_ID', 'FLAG_TYPE', 'MERGE_SRC', 'MERGE_SRC_ID'):
        if col in df.columns:
            del(df[col])

    df.columns = [re.sub('^MERGE_', 'HAP_', val) for val in df.columns]
    df.columns = ['HAP' if val == 'HAP_SAMPLES' else val for val in df.columns]

    # Get values per haplotype
    for col in ('HAP', 'HAP_VARIANTS', 'HAP_RO', 'HAP_SZRO', 'HAP_OFFSET', 'HAP_OFFSZ', 'HAP_MATCH'):
        if col in df.columns:
            df[col] = df[col].apply(lambda val: ';'.join(val.split(',')))

    # Pack fields with values from all haplotypes
    df_dict = {
        key: val for key, val in zip(hap_list, bed_list)
    }

    if df.shape[0] > 0:
        for col in ('QRY_REGION', 'QRY_STRAND', 'CI', 'ALIGN_INDEX', 'CALL_SOURCE', 'RGN_REF_INNER', 'RGN_QRY_INNER', 'COV_MEAN', 'COV_PROP', 'COV_QRY'):
            if col in df.columns:
                df[col] = val_per_hap(df, df_dict, col)

    # Load mapped regions
    map_tree_list = list()

    for index in range(n_hap):
        map_tree = collections.defaultdict(intervaltree.IntervalTree)

        for index, row in pd.read_csv(callable_list[index], sep='\t').iterrows():
            map_tree[row['#CHROM']][row['POS']:row['END']] = True

        map_tree_list.append(map_tree)

    # Get genotypes setting no-call for non-mappable regions
    if df.shape[0] > 0:
        df['GT'] = pd.concat(
            [
                df.apply(get_gt, hap=hap_list[index], map_tree=map_tree_list[index], axis=1)
                    for index in range(n_hap)
            ],
            axis=1
        ).apply(lambda vals: '|'.join(vals), axis=1)
    else:
        df['GT'] = ''

    # Return merged BED
    return df


def get_merge_params(
        svtype: str,
        pav_params: pavconfig.ConfigParams
) -> str:
    """
    Get merging parameters.

    :param svtype: SV type.
    :param pav_params: PAV parameters.

    :return: An SV-Pop merge definition string describing how variants should be intersected.
    """

    svtype = svtype.lower().strip()

    if svtype in {'ins', 'del', 'insdel'}:
        return pav_params.merge_insdel

    if svtype == 'inv':
        return pav_params.merge_inv

    if svtype == 'snv':
        return pav_params.merge_snv

    if svtype == 'cpx':
        return pav_params.merge_cpx

    if svtype == 'dup':
        return pav_params.merge_dup

    raise RuntimeError(f'Unknown SV type: {svtype}')


def apply_trim_filter(
    df: pd.DataFrame,
    filter_dict: dict,
    trim_tree: intervaltree.IntervalTree
):
    """
    Apply trim filters to a filter dictionary.

    :param df: Variant call table.
    :param filter_dict: Filter dictionary.
    :param trim_tree: Interval tree of trimmed regions.
    """

    pos_arry = df[['ALIGN_INDEX', 'QRY_POS', 'QRY_END']].values
    id_arr = df['ID'].values

    for arr_index in range(pos_arry.shape[0]):
        index, qry_pos, qry_end = pos_arry[arr_index]
        var_id = id_arr[arr_index]

        for match in trim_tree[(index, qry_pos):(index, qry_end)]:
            filter_dict[var_id].add(match.data)

def read_filter_tree(filter_list, default_none=True):
    """
    Read a list of BED files into an interval tree.

    :param filter_list: List of BED file names (type `list`) or a semi-colon separated list of bed file names
        (type `str`).
    :param default_none: If there are no filters, return `None` if `True`, otherwise retrun an empty interval tree.
    """

    if isinstance(filter_list, str):
        filter_list = [filename for val in filter_list.split(';') if (filename := val.strip()) != '']
    elif isinstance(filter_list, list):
        filter_list = [filename for val in filter_list if (filename := val.strip()) != '']
    elif filter_list is None:
        filter_list = []
    else:
        raise RuntimeError(f'Expected filter_list to be a list, string, or None, received {type(filter_list)}')

    if len(filter_list) == 0:
        if default_none:
            return None
        else:
            return collections.defaultdict(intervaltree.IntervalTree)

    filter_tree = collections.defaultdict(intervaltree.IntervalTree)

    for filter_filename in filter_list:
        df_filter = pd.read_csv(filter_filename, sep='\t', header=None, comment='#', usecols=(0, 1, 2))
        df_filter.columns = ['#CHROM', 'POS', 'END']

        for index, row in df_filter.iterrows():
            filter_tree[row['#CHROM']][row['POS']:row['END']] = True

    return filter_tree

def read_trim_regions(
    df_none: pd.DataFrame,
    df_qry: pd.DataFrame,
    df_qryref: pd.DataFrame
) -> intervaltree.IntervalTree:
    """
    Get an intervaltree describing trimmed alignment filters.

    :param df_none: Alignment table with no trimming.
    :param df_qry: Alignment table after query trimming.
    :param df_qryref: Alignment table after query and reference trimming.
    """

    trim_tree = intervaltree.IntervalTree()

    for filter_str in ('TRIMQRY', 'TRIMREF'):

        # Choose left (pre-trim) and right (post-trim) alignment records
        if filter_str == 'TRIMQRY':
            df_l = df_none
            df_r = df_qry
        elif filter_str == 'TRIMREF':
            df_l = df_qry
            df_r = df_qryref
        else:
            raise RuntimeError(f'Expected filter_str to be "TRIMQRY" or "TRIMQRYREF", received {filter_str}')

        # Get array of coordinates (start_l, pos_r, end_l, end_r), -1 on r coordinates means the whole record was trimmed
        coord_arr = df_l[['INDEX', 'QRY_POS', 'QRY_END']].set_index('INDEX').join(
            df_r[['INDEX', 'QRY_POS', 'QRY_END']].set_index('INDEX'),
            how='left', lsuffix='_L', rsuffix='_R'
        ).reset_index(drop=False)[
            ['QRY_POS_L', 'QRY_POS_R', 'QRY_END_L', 'QRY_END_R', 'INDEX']
        ].fillna(-1).astype(int).values

        # Fill tree with trimmed regions
        for index in range(coord_arr.shape[0]):
            align_index = coord_arr[index, 4]

            if coord_arr[index, 1] >= 0:
                if coord_arr[index, 1] > coord_arr[index, 0]:
                    # Left side was trimmed
                    trim_tree[(align_index, coord_arr[index, 0]):(align_index, coord_arr[index, 1])] = filter_str

                if coord_arr[index, 3] < coord_arr[index, 2]:
                    # Right side was trimmed
                    trim_tree[(align_index, coord_arr[index, 3]):(align_index, coord_arr[index, 2])] = filter_str

            else:
                # Right record was  fully trimmed
                trim_tree[(align_index, coord_arr[index, 0]):(align_index, coord_arr[index, 2])] = filter_str

    return trim_tree

def read_inner_tree(
        seg_filename: str,
        trim_tree: intervaltree.IntervalTree,
        purge_filter_tree: collections.defaultdict=None
) -> intervaltree.IntervalTree:
    """
    Get a dictionary of alignment records internal to larger variants.

    :param seg_filename: Segment table filename.
    :param trim_tree: Filter tree for trimmed alignments.
    :param purge_filter_tree: Filter tree for alignments to purge.
    """

    inner_tree = intervaltree.IntervalTree()

    # Read segment table for large variants
    df_seg = pd.read_csv(
        seg_filename,
        sep='\t',
        usecols=['ID', '#CHROM', 'POS', 'END', 'FILTER', 'INDEX', 'QRY_POS', 'QRY_END', 'IS_ALIGNED', 'IS_ANCHOR']
    )

    df_seg = df_seg.loc[
        df_seg['IS_ALIGNED']
    ]

    # Separate anchors and aligned variant segments
    df_anchor = df_seg.loc[df_seg['IS_ANCHOR']]
    df_seg = df_seg.loc[~df_seg['IS_ANCHOR']]

    # Do not allow duplicated segments
    dup_index = {
        align_index for align_index, count in
            zip(*np.unique(df_seg['INDEX'], return_counts=True))
                if count > 1
    }

    if dup_index:
        n = len(dup_index)
        index_list = ', '.join(sorted(dup_index)[:3]) + ('...' if n > 3 else '')
        raise RuntimeError(f'Found {n} duplicate alignment indices in the segment table: {index_list}')

    # Set filter
    if 'FILTER' not in df_seg.columns:
        df_seg['FILTER'] = 'PASS'

    df_seg['FILTER'] = df_seg['FILTER'].fillna('PASS').astype(str)

    # Set inner dict
    for index, row in df_seg.iterrows():
        inner_tree[
            (row['INDEX'], row['QRY_POS']): (row['INDEX'], row['QRY_END'])
        ] = (
            row['ID'],
            {val.strip() for val in row['FILTER'].split(',')} - {'PASS', ''}
        )

    # Update purge filter
    update_purge_filter = purge_filter_tree is not None

    for var_id, subdf in df_anchor.groupby('ID'):
        if subdf.shape[0] != 2:
            continue # Shouldn't occur, all LG-SVs should have two anchors

        row_l, row_r = subdf.iloc[0], subdf.iloc[1]

        # Update purge filter if there is a gap between anchors
        if update_purge_filter and row_r['POS'] > row_l['END']:
            purge_filter_tree[row_l['#CHROM']][row_l['END']:row_r['POS']] = var_id

        # Update inner dict if the anchors overlap (tandem DUP over anchors).
        # Add regions that would be trimmed by reference trimming
        if row_r['POS'] < row_l['END']:

            filter_set = (set(row_l['FILTER'].split(',')) | set(row_r['FILTER'].split(','))) - {'PASS', ''}

            for region in trim_tree[
                (row_l['INDEX'], row_l['QRY_POS']):(row_l['INDEX'], row_l['QRY_END'])
            ] | trim_tree[
                (row_r['INDEX'], row_r['QRY_POS']):(row_r['INDEX'], row_r['QRY_END'])
            ]:
                if region.data != 'TRIMREF':
                    continue

                align_index = region.begin[0]
                qry_pos = region.begin[1]
                qry_end = region.end[1]

                inner_tree[
                    (align_index, region.begin[1]):(align_index, region.end[1])
                ] = (var_id, filter_set)

    return inner_tree

def apply_inner_filter(
    df: pd.DataFrame,
    filter_dict: dict,
    inner_tree: intervaltree.IntervalTree,
    inner_dict: dict,
    filter_str: str='INNER'
) -> None:
    """
    Apply filter for inner variants.

    :param df: Variant call table.
    :param filter_dict: Filter dictionary.
    :param inner_tree: Interval tree of inner regions.
    :param inner_dict: Dict keyed by variant IDs containing sets of FILTER strings.
    :param filter_str: String to atd to `filter_dict` for inner variants.
    """

    pos_arry = df[['ALIGN_INDEX', 'QRY_POS', 'QRY_END']].values
    id_arr = df['ID'].values

    for arr_index in range(pos_arry.shape[0]):
        index, qry_pos, qry_end = pos_arry[arr_index]
        var_id = id_arr[arr_index]

        for match in inner_tree[(index, qry_pos):(index, qry_end)]:
            inner_dict[var_id].add(match.data[0])
            filter_dict[var_id].update(match.data[1])

            filter_dict[var_id].add(filter_str)

def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """
    Convert a Pandas DataFrame to Polars. Handle types for known columns and set missing values appropriately to
    null in the Polars DataFrame.

    :param df: Pandas DataFrame to convert.

    :return: Polars DataFrame
    """

    col_list = list()

    for col_name in df.columns:

        try:
            col = df[col_name].copy()
            dtype = PD_COL_TYPES.get(col_name, np.str_)

            null_vals = col.isnull()

            if np.issubdtype(dtype, np.floating):
                null_vals |= col == ''
                col[null_vals] = np.nan

            elif np.issubdtype(dtype, np.integer):
                null_vals |= col == ''
                col[null_vals] = 0

            elif np.issubdtype(dtype, np.bool):
                null_vals |= col == ''
                col[null_vals] = False

                if np.issubdtype(col.dtype, np.str_):
                    col = col.apply(util.as_bool)

            elif np.issubdtype(dtype, np.str_):
                if not np.issubdtype(col.dtype, np.str_):
                    col = col.astype(str)

                col[null_vals] = ''

            else:
                raise RuntimeError(f'Unhandled dtype: {dtype}')

            col = pl.from_pandas(
                col.astype(dtype)
            )

            if null_vals.any():
                col.to_frame(
                ).with_row_index(
                    '_index'
                ).join(
                    pl.from_pandas(null_vals).rename('_is_null').to_frame().with_row_index('_index'),
                    on='_index',
                    how='left'
                ).select(
                    pl.when(pl.col._is_null).then(None).otherwise(pl.col(col.name)).alias(col.name)
                ).to_series()

            col_list.append(col)

        except Exception as e:
            raise RuntimeError(f'Failed to convert column {col_name} to polars: {e}')

    try:
        return pl.DataFrame(col_list)

    except Exception as e:
        raise RuntimeError(f'Failed transforming converted columns to polars dataframe: {e}')
