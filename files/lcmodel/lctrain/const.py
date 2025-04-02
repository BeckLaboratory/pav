"""
LC model training constants.
"""

import pavlib

#
# Default parameters
#

# Default hold-out test size
DEFAULT_TEST_SIZE = 20000

# Default number of cross-validation folds
DEFAULT_K_FOLD = 4

# Default alignment score proportion for initially confident alignments for feature development.
DEFAULT_SCORE_PROP_CONF = 0.85

# Default alignment score proportion for rescuing alignments from low-confidence clusters (must also match the rescue length).
DEFAULT_SCORE_PROP_RESCUE = 0.95

# Default alignment score mismatch proportion for initially confident alignments for feature development.
DEFAULT_SCORE_MM_PROP_CONF = None

# Default alignment score mismatch proportion for rescuing alignments from low-confidence clusters (must also match the rescue length).
DEFAULT_SCORE_MM_PROP_RESCUE = None

# Default length for rescuing alignments from low-confidence clusters (must also match score proportion)
DEFAULT_RESCUE_LENGTH = 10000

# Default flank for merging adjacent low-confidence alignments
DEFAULT_MERGE_FLANK = 2000

# Default score model definition (saved here so the features stage can find it)
DEFAULT_SCORE_MODEL = pavlib.align.score.DEFAULT_ALIGN_SCORE_MODEL
