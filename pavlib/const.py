"""
Program constants
"""

# Version constants
VERSION_MAJOR = 3     # Major change or a large number of minor changes
VERSION_MINOR = 0     # A significant change it PAV or a large number of small incremental changes
VERSION_DEV = 0       # Small changes, usually bug fixes
VERSION_PATCH = 1     # Development and test versions, not for releases


# VERSION_PATCH is "None" for releases or set if:
#
# 1) Testing fixes before a release.
#
# 2) In rare cases when an older version of PAV is patched, the patch version is set. Changes to those versions are
# not synced with the main branch, but may contain fixes applied to later and replicated into an earlier branch.
# This is rarely used and only in special circumstances when a very large project is using a specific version but needs
# bug without significant changes to PAV.


def get_version_string():
    """
    Get PAV version as a string.

    :return: PAV version string.
    """

    return f'{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_DEV}' + (f'.{VERSION_PATCH}' if VERSION_PATCH is not None else '')


#
# Call parameters
#

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

DEFAULT_MIN_ANCHOR_SCORE = '50bp'

DEFAULT_LG_OFF_GAP_MULT = 1.5

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


#
# Return Codes
#

# When scripts/density.py fails to find an inversion but not because of an error, this value is returned to indicate
# that the pipeline should go ahead without crashing. Return codes other than 0 and ERR_INV_FAIL indicate a problem or
# that should be corrected.
ERR_INV_FAIL = 125
