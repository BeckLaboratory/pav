"""Program constants."""

__all__ = [
    'FILTER_REASON',
    'DEFAULT_MIN_ANCHOR_SCORE',
    'MERGE_PARAM_INSDEL',
    'MERGE_PARAM_INV',
    'MERGE_PARAM_SNV',
    'MERGE_PARAM_DEFAULT',
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
# Filters
#

# Explanations for filter codes
FILTER_REASON = {
    'LCALIGN': 'Variant inside a low-confidence alignment record',
    'ALIGN': 'Variant inside an alignment record that had a filtered flag (matches 0x700 in alignment flags) or did '
             'not meet a minimum MAPQ threshold',
    'DISCORD': 'Discordant with another variant (i.e. small variants inside a deletion)',
    'INNER': 'Part of a larger variant call (i.e. SNVs and indels inside a duplication or complex event)',
    'DERIVED': 'A noncanonical variant form derived from another (i.e. DUP derived from an INS variant or DELs and DUPs from complex events)',
    'VARLEN': 'Variant size out of set bounds (sizes set in the PAV config file)',
    'TRIMREF': 'Alignment trimming in reference coordinates removed variant',
    'TRIMQRY': 'Alignment trimming in query coordinates removed variant',
}
"""Explanation of filter codes found in alignment and variant records."""


#
# Call parameters
#

DEFAULT_MIN_ANCHOR_SCORE: str | float = '50bp'
"""Minimum score for anchoring sites in large alignment-truncating SVs (LGSV module)"""

# Default merge for INS/DEL/INV
MERGE_PARAM_INSDEL: str = 'nr::ro(0.5):szro(0.5,200,2):match'
"""Default merge for INS/DEL."""

MERGE_PARAM_INV: str = 'nr::ro(0.2)'
"""Default merge for INV."""

MERGE_PARAM_SNV: str = 'nrsnv::exact'
"""Default merge for SNV."""

MERGE_PARAM_DEFAULT: dict[str, str] = {
    'ins': MERGE_PARAM_INSDEL,
    'del': MERGE_PARAM_INSDEL,
    'insdel': MERGE_PARAM_INSDEL,
    'inv': MERGE_PARAM_INV,
    'snv': MERGE_PARAM_SNV
}
"""Default merge parameters by variant type."""

DEFAULT_LG_OFF_GAP_MULT: float = 4.5

DEFAULT_LG_GAP_SCALE: float = 0.2

DEFAULT_LG_SMOOTH_SEGMENTS: float = 0.05
"""
Smoothing factor as a minimum proportion of variant length to retain. For example, at 0.05,
any segment smaller than 5% of the total SV length is smoothed out (assuming approximate colinearity)
simplifying the annotated structure. Variant calls retain the original segments, so while this creates an
approximation of the structure for the call, the full structure is not lost.
"""


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
