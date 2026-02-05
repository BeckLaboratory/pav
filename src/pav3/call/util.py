"""Variant calling utilities."""

__all__ = [
    'filter_trim'
]

import polars as pl
import polars.selectors as cs

COMPL_TR_FROM = [
    'a', 'A', 'c', 'C', 'g', 'G', 't', 'T',  # 1-base
    'r', 'R', 'm', 'M', 'w', 'W', 's', 'S', 'y', 'Y', 'k', 'K',  # 2-base
    'v', 'V', 'h', 'H', 'd', 'D', 'b', 'B',  # 3-base
    'n', 'N',  # 4-base
    'u', 'U',  # U to T
]
"""Base complement table on IUPAC notation, source base. Complement is in `COMPL_TR_TO` at the same index."""

COMPL_TR_TO = [
    't', 'T', 'g', 'G', 'c', 'C', 'a', 'A',  # 1-base
    'y', 'Y', 'k', 'K', 'w', 'W', 's', 'S', 'r', 'R', 'm', 'M',  # 2-base
    'b', 'B', 'd', 'D', 'h', 'H', 'v', 'V',  # 3-base
    'n', 'N',  # 4-base
    't', 'T',  # U to T
]
"""Base complement table on IUPAC notation, complement base. Source base is in `COMPL_TR_FROM` at the same index."""

# # Code for generating base complement table:
#
# base_to_code = {
#     ('a',): 'a',
#     ('c',): 'c',
#     ('g',): 'g',
#     ('t',): 't',
#
#     ('a', 'g',): 'r',
#     ('a', 'c',): 'm',
#     ('a', 't',): 'w',
#     ('c', 'g',): 's',
#     ('c', 't',): 'y',
#     ('g', 't',): 'k',
#
#     ('a', 'c', 'g',): 'v',
#     ('a', 'c', 't',): 'h',
#     ('a', 'g', 't',): 'd',
#     ('c', 'g', 't',): 'b',
#
#     ('a', 'c', 'g', 't',): 'n',
# }
#
# compl_tr = {
#     'a': 't',
#     't': 'a',
#     'c': 'g',
#     'g': 'c',
# }
#
# tr_list = []
#
# for key, value in base_to_code.items():
#     compl = base_to_code[
#         tuple(sorted((compl_tr[b] for b in key)))
#     ]
#
#     tr_list.append((value, compl))
#     tr_list.append((value.upper(), compl.upper()))
#
# tr_list.append(('u', 't'))
# tr_list.append(('U', 'T'))
#
# tr_from = [v[0] for v in tr_list]
# tr_to = [v[1] for v in tr_list]


# def filter_trim(
#         df: pl.DataFrame | pl.LazyFrame,
#         df_align: pl.DataFrame | pl.LazyFrame,
# ) -> pl.LazyFrame:
#     """Filter a variant table by trimmed alignments.
#
#     Retain only variants that are not fully within non-trimmed regions.
#
#     :param df: Variant table.
#     :param df_align: Alignment table this variant table was generated from. The align_index field in the variant table
#         must match the alignment record the variant was generated from.
#
#     :returns: Filtered variant table.
#     """
#     if not isinstance(df, pl.LazyFrame):
#         df = df.lazy()
#
#     if not isinstance(df_align, pl.LazyFrame):
#         df_align = df_align.lazy()
#
#     return (
#         df
#         .with_columns(
#             pl.col('align_source').arr.get(0).alias('align_index'),
#         )
#         .join(
#             df_align.select(
#                 pl.col('align_index'),
#                 pl.col('qry_pos').alias('_qry_pos_r'),
#                 pl.col('qry_end').alias('_qry_end_r'),
#             ),
#             on='align_index',
#             how='left'
#         )
#         .filter(
#             pl.col('_qry_pos_r').is_not_null() & (
#                 (pl.col('qry_end') < pl.col('_qry_pos_r')) | (pl.col('qry_pos') > pl.col('_qry_end_r'))
#             )
#         )
#         .drop(
#             cs.starts_with('_')
#         )
#     )
