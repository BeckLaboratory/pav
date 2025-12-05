"""Program constants."""

from typing import Any

__all__ = [
    'FILTER_REASON',
    'DEFAULT_MIN_ANCHOR_SCORE',
    'DEFAULT_LG_OFF_GAP_MULT',
    'DEFAULT_LG_GAP_SCALE',
    'DEFAULT_LG_SMOOTH_SEGMENTS',
    'INV_K_SIZE',
    'INV_INIT_EXPAND',
    'INV_EXPAND_FACTOR',
    'INV_REGION_LIMIT',
    'INV_MIN_KMERS',
    'INV_MIN_INV_KMER_RUN',
    'INV_MIN_QRY_REF_PROP',
    'INV_MIN_EXPAND_COUNT',
    'INV_MAX_REF_KMER_COUNT',
    'INV_KDE_BANDWIDTH',
    'INV_KDE_TRUNC_Z',
    'INV_REPEAT_MATCH_PROP',
    'INV_KDE_FUNC',
]


#
# Call parameters
#

DEFAULT_MIN_ANCHOR_SCORE: str | float = '50bp'
"""Minimum score for anchoring sites in large alignment-truncating SVs (LGSV module)"""

DEFAULT_LG_OFF_GAP_MULT: float = 4.5
"""
For large variantns, multiply penalties for gaps inconsistent with their variant types (e.g. DEL around an INS) by
this factor. A higher number will require cleaner variants for simple variant types (INS, DEL, INV) and push noisier
breakpoints into complex (CPX) events.
"""

DEFAULT_LG_GAP_SCALE: float = 0.2
"""
Controls how strong anchoring alignments need to be to support large alignment-truncatincg variants. The gap is penalty
calculated on the number of query bases between the anchors and multiplied by this factor, and if the magnitude of the
penalty exceeds the magnitude of the anchor score, the two alignments are not allowed to anchor a variant. A number
less than 1.0 will allow smaller anchors to support larger variants, and a number greater than 1.0 will require stronger
anchors. 
"""


DEFAULT_LG_SMOOTH_SEGMENTS: float = 0.05
"""
Smoothing factor as a minimum proportion of variant length to retain. For example, at 0.05,
any segment smaller than 5% of the total SV length is smoothed out (assuming approximate colinearity)
simplifying the annotated structure. Variant calls retain the original segments, so while this creates an
approximation of the structure for the call, the full structure is not lost.
"""


#
# Align parameters
#

DEFAULT_ALIGN_TRIM_MAX_DEPTH = 20
"""Default maximum depth for alignments before applying the DEPTH filter."""


#
# Inversion parameters
#

INV_K_SIZE: int = 31
"""K-mer size for inversion calling."""

INV_INIT_EXPAND: int = 4000
"""Expand the flagged region by this much before starting."""

INV_EXPAND_FACTOR: float = 1.5
"""Expand by this factor while searching"""

INV_REGION_LIMIT: int = 1000000
"""Maximum region size"""

INV_MIN_KMERS: int = 1000
"""
Minimum number of k-mers with a distinct state (sum of FWD, FWDREV, and REV). Stop if the number of k-mers is less after
filtering uninformative and high-count k-mers.
"""

INV_MIN_INV_KMER_RUN: int = 100
"""States must have a continuous run of this many strictly inverted k-mers"""

INV_MIN_QRY_REF_PROP: float = 0.6
"""
The query and reference region sizes must be within this factor (reciprocal) or the event is likely unbalanced
(INS or DEL) and would already be in the callset
"""

INV_MIN_EXPAND_COUNT: int = 3
"""
The default number of region expansions to try (including the initial expansion) and finding only fwd k-mer states
after smoothing before giving up on the region.
"""

INV_MAX_REF_KMER_COUNT: int = 10
"""If canonical reference k-mers have a higher count than this, they are discarded"""

INV_KDE_BANDWIDTH: float = 100.0
"""Convolution KDE bandwidth for"""

INV_KDE_TRUNC_Z: float = 3.0
"""Convolution KDE truncated normal at Z (in standard normal, scaled by bandwidth)"""

INV_REPEAT_MATCH_PROP: float = 0.15
"""When scoring INV structures, give a bonus to inverted repeats that are similar in size scaled by this factor"""

INV_KDE_FUNC: float | str = 'auto'
"""Inversion convolution method.

Convolution method. "fft" is a Fast-Fourier Transform, "conv" is a standard linear convolution. "auto" uses "fft" if
available and falls back to "conv" otherwise.
"""

DEFAULT_MERGE_PARAMS: dict[str, list[dict[str, Any]]] = {
    'insdel': [
        {
            'ro_min': 0.5,
            'match_prop_min': 0.8,
            'match_vartype': True,
        },
        {
            'size_ro_min': 0.8,
            'offset_max': 200,
            'match_prop_min': 0.8,
            'match_vartype': True,
        },
        {
            'offset_prop_max': 2.0,
            'size_ro_min': 0.8,
            'match_prop_min': 0.8,
            'match_vartype': True,
        }
    ],
    'inv': [
        {
            'ro_min': 0.2,
        },
    ],
    'snv': [
        {
            'offset_max': 0,
            'match_ref': True,
            'match_alt': True,
        },
    ],
    'cpx': [
        {
            'ro_min': 0.5,
            'seg_ro_min': 0.5,
            'match_prop_min': 0.8,
        },
    ]
}
"""Default parameters for merging haplotypes."""


#
# Filters
#

# Explanations for filter codes
FILTER_REASON: dict[str, str] = {
    'LCALIGN': 'Variant inside a low-confidence alignment record',
    'ALIGN': 'Variant inside an alignment record that had a filtered flag (matches 0x700 in alignment flags) or did '
             'not meet a minimum MAPQ threshold',
    'DISCORD': 'Discordant with another variant (i.e. small variants inside a deletion)',
    'INNER': 'Part of a larger variant call (i.e. SNVs and indels inside a duplication or complex event)',
    'DERIVED': 'A noncanonical variant form derived from another (i.e. DUP derived from an INS variant or DELs and DUPs from complex events)',
    'VARLEN': 'Variant size out of set bounds (sizes set in the PAV config file)',
    'TRIMREF': 'Alignment trimming in reference coordinates removed variant',
    'TRIMQRY': 'Alignment trimming in query coordinates removed variant',
    'DEPTH': f'Region was covered by many redundant alignments '
             f'({DEFAULT_ALIGN_TRIM_MAX_DEPTH} by default, tunable with parameter align_trim_max_depth)'
}
"""Explanation of filter codes found in alignment and variant records."""