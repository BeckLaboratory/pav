"""Variant calling utilities."""

__all__ = [
    'COMPL_TR_FROM',
    'COMPL_TR_TO',
]

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
