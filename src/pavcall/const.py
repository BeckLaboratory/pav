"""
Program constants
"""

#
# Filters
#

# Explanations for filter codes
FILTER_REASON = {
    'LCALIGN': 'Variant inside a low-confidence alignment record',
    'ALIGN': 'Variant inside an alignment record that had a filtered flag (matches 0x700 in alignment flags) or did not meet a minimum MAPQ threshold',
    'QRY_FILTER': 'Query filter region (regions provided to PAV at runtime)',
    'PURGE': 'Inside larger variant',  # Previously "COMPOUND"
    'INNER': 'Part of a larger variant call (i.e. a variant inside of a larger duplication)',
    'SVLEN': 'Variant size out of set bounds (sizes set in the PAV config file)',
    'TRIMREF': 'Alignment trimming in reference coordinates removed variant',
    'TRIMQRY': 'Alignment trimming in query coordinates removed variant'
}
"""Explanation of filter codes found in alignment and variant records."""


#
# Call parameters
#

DEFAULT_MIN_ANCHOR_SCORE = '50bp'
"""Minimum score for anchoring sites in large alignment-truncating SVs (LGSV module)"""

# Default merge for INS/DEL/INV
MERGE_PARAM_INSDEL = 'nr::ro(0.5):szro(0.5,200,2):match'
MERGE_PARAM_INV = 'nr::ro(0.2)'
MERGE_PARAM_SNV = 'nrsnv::exact'

MERGE_PART_COUNT = 10

MERGE_PARAM_DEFAULT = {
    'ins': MERGE_PARAM_INSDEL,
    'del': MERGE_PARAM_INSDEL,
    'inv': MERGE_PARAM_INV,
    'snv': MERGE_PARAM_SNV
}


DEFAULT_LG_OFF_GAP_MULT = 4.5

DEFAULT_LG_GAP_SCALE = 0.2

DEFAULT_LG_SMOOTH_SEGMENTS = 0.05


#
# Inversion parameters
#

INV_K_SIZE = 31
INV_INIT_EXPAND = 4000        # Expand the flagged region by this much before starting.
INV_EXPAND_FACTOR = 1.5       # Expand by this factor while searching

INV_REGION_LIMIT = 1000000       # Maximum region size

INV_MIN_KMERS = 1000          # Minimum number of k-mers with a distinct state (sum of FWD, FWDREV, and REV). Stop if
                              # the number of k-mers is less after filtering uninformative and high-count k-mers.

INV_MIN_INV_KMER_RUN = 100    # States must have a continuous run of this many strictly inverted k-mers

INV_MIN_QRY_REF_PROP = 0.6    # The query and reference region sizes must be within this factor (reciprocal) or the event
                              # is likely unbalanced (INS or DEL) and would already be in the callset

INV_MIN_EXPAND_COUNT = 3      # The default number of region expansions to try (including the initial expansion) and
                              # finding only fwd k-mer states after smoothing before giving up on the region.

INV_MAX_REF_KMER_COUNT = 10   # If canonical reference k-mers have a higher count than this, they are discarded

INV_KDE_BANDWIDTH = 100.0     # Convolution KDE bandwidth for

INV_KDE_TRUNC_Z = 3.0         # Convolution KDE truncated normal at Z (in standard normal, scaled by bandwidth)

INV_REPEAT_MATCH_PROP = 0.15  # When scoring INV structures, give a bonus to inverted repeats that are similar in size scaled by this factor

INV_KDE_FUNC = 'auto'         # Convolution method. "fft" is a Fast-Fourier Transform, "conv" is a standard linear convolution.
                              #  "auto" uses "fft" if available and falls back to "conv" otherwise.
