"""
PAV configuration parameter definitions and utilities.

A consistent interface for creating and accessing configuration parameters is provided by the `ConfigParams` class.
"""

__all__ = [
    'DEFAULT_ALIGNER',
    'DEFAULT_ALIGNER_PARAMS',
    'NAMED_ALIGNER_PARAMS',
    'KNOWN_ALIGNERS',
    'PavParams',
    'format_config_md',
    'get_align_params',
]

from dataclasses import dataclass
import polars as pl
import sys
import textwrap
from typing import Any, Optional, TextIO

from . import const

from .align.score import DEFAULT_ALIGN_SCORE_MODEL
from .util import as_bool

DEFAULT_ALIGNER: str = 'minimap2'
"""Default alignment program."""

DEFAULT_ALIGNER_PARAMS: dict[str, str] = {
    'minimap2': '-x asm20',
    'lra': '',
}
"""Default alignment parameters by aligner."""

NAMED_ALIGNER_PARAMS: dict[str, dict[str, str]] = {
    'pav2': {
        'minimap2': '-x asm20 -m 10000 -z 10000,50 -r 50000 --end-bonus=100 -O 5,56 -E 4,1 -B 5',
        'lra': ''
    }
}
"""A set of named alignment parameters to support string names for parameter sets."""

KNOWN_ALIGNERS: list[str] = sorted(DEFAULT_ALIGNER_PARAMS.keys())
"""List of known aligners."""


#
# Configuration parameters
#

class PavParams:
    """Manage pipeline configurations."""

    def __init__(
        self,
        asm_name: Optional[str] = None,
        pav_config: Optional[dict] = None,
        asm_table: Optional[pl.DataFrame] = None,
        verbose: Optional[bool] = None
    ) -> None:
        """Initialize PavParams.

        :param asm_name: Assembly name.
        :param pav_config: PAV configuration.
        :param asm_table: Assembly table.
        :param verbose: Verbose output.
        """
        self._asm_name = asm_name
        self._pav_config = dict(pav_config) if pav_config is not None else dict()
        self._asm_table = asm_table

        self._set_config_override_dict()

        if verbose is None:
            self.verbose = CONFIG_PARAM_DICT['verbose'].get_value(
                self._config_override.get('verbose', self._pav_config.get('verbose', None))
            )
        else:
            self.verbose = verbose

    def __getattr__(self, key):
        """Get a configuration parameter, set if needed.

        Caches value to self.
        """
        if key not in CONFIG_PARAM_DICT.keys():
            raise KeyError(f'Unknown configuration parameter "{key}"')

        if key == 'align_params':
            setattr(
                self,
                key,
                get_align_params(
                    self.aligner,
                    CONFIG_PARAM_DICT[key].get_value(
                        self._config_override.get(key, self._pav_config.get(key, None))
                    )
                )
            )

        else:
            setattr(
                self,
                key,
                CONFIG_PARAM_DICT[key].get_value(
                    self._config_override.get(key, self._pav_config.get(key, None))
                )
            )

        val = getattr(self, key)

        if self.verbose:
            print(f'Config: {key} = {val} [{str(self)}]')

        return val

    def _set_config_override_dict(self) -> None:
        """Set a dictionary of overridden parameters using the CONFIG column of the assembly table."""
        # Init config override
        self._config_override = dict()

        # Get override config string
        if self._asm_table is None or self._asm_name is None or self._asm_name not in self._asm_table['name']:
            return

        config_string = (
            self._asm_table
            .filter(pl.col('name') == self._asm_name)
            .select('config')
            .item(0, 0)
        )

        config_string = config_string.strip() if config_string is not None else None

        if not config_string:
            return

        # Process each config directive
        tok_list = config_string.split(';')

        for tok in tok_list:

            # Check tok
            tok = tok.strip()

            if not tok:
                continue

            if '=' not in tok:
                raise RuntimeError(
                    f'Cannot get assembly configuration: Missing "=" in config token {tok}: {config_string}'
                )

            # Get attribute and value
            key, val = tok.split('=', 1)

            key = key.strip()
            val = val.strip()

            if not key:
                raise RuntimeError(
                    f'Cannot get assembly configuration: Missing key (key=value) in config token {tok}: '
                    f'{config_string}'
                )

            if not val:
                raise RuntimeError(
                    f'Cannot get assembly configuration: Missing value (key=value) in config token {tok}: '
                    f'{config_string}'
                )

            # Set value
            self._config_override[key] = val

    def get_aligner_index_input(self):
        """Get a list of index files needed by an aligner.

        :returns: Alignment index files.
        """
        # Check parameters
        aligner = self.aligner

        # Return list of input file (FASTA file is first)
        if aligner == 'minimap2':
            return []

        # if aligner == 'lra':
        #     return [
        #         'data/ref/ref.fa.gz.gli',
        #         'data/ref/ref.fa.gz.mms'
        #     ]

        raise RuntimeError(f'Unknown aligner: {aligner}')

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            f'ConfigParams('
            f'asm_name={self._asm_name}, '
            f'config={'CONFIG' if self._pav_config is not None else 'None'}, a'
            f'sm_table={'ASM_TABLE' if self._asm_table is not None else 'None'})'
        )


@dataclass(frozen=True)
class _ConfigParamElement(object):
    """A configuration parameter object.

    Minimum and maximum values can be either a single value to check or a tuple of (value, inclusive/exclusive). If
    it is a single value, then the min/max is inclusive. If it is a tuple, then min/max is inclusive if the second
    element of the tuple is `True` and exclusive if it is `False`.

    :param name: Parameter name.
    :param val_type: Type of parameter as a string.
    :param default: Default value.
    :param min: Minimum value if not `None`.
    :param max: Maximum value if not `None`.
    :param allowed: Set of allowed values if not `None`.
    :param to_lower: String value is converted to lower case if `True`. Only valid if `val_type` is `str`.
    :param fail_none: If `True`, fail if a parameter value is `None`, otherwise, return the default value.
    :param description: Description of the parameter.
    :param is_null: `True` if the parameter is included for documentation purposes, but parameter processing is handled
        outside this class. Alignment parameters must be adjusted for the aligner, and so it is not processed here,
        however, the alignment ConfigParam objects are included to simplify parameter documentation.
    :param advanced: `True` if the parameter is an advanced option and should not be shown in brief documentation.
    """

    name: str
    val_type: str
    default: Optional[Any] = None
    min: Optional[Any] = None
    max: Optional[Any] = None
    allowed: set[Any] = None
    to_lower: bool = False
    fail_none: bool = False
    description: str = None
    is_null: bool = False
    advanced: bool = False

    def __post_init__(self) -> None:
        """Check attributes."""
        # Check name
        if self.name is None or not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('name is missing or empty')

        object.__setattr__(self, 'name', self.name.strip().lower())

        # Check type
        if self.val_type is None or not isinstance(self.val_type, str) or not self.val_type.strip():
            raise ValueError('Type is missing or empty')

        object.__setattr__(self, 'val_type', self.val_type.strip().lower())

        if self.val_type not in {'int', 'float', 'bool', 'str'}:
            raise ValueError(f'Unrecognized parameter type: {self.val_type}')

        # Check allowed values
        if self.allowed is not None:
            if not isinstance(self.allowed, set):
                raise ValueError(f'Allowed values must be a set: {type(self.allowed)}')

            object.__setattr__(self, 'allowed', self.allowed.copy())

        # Check min/max
        if self.min is not None or self.max is not None:
            if self.val_type not in {'int', 'float'}:
                raise ValueError(f'min/max must be int/float: {type(self.val_type)}')

        # Check to_lower
        if self.to_lower not in {True, False}:
            raise ValueError(f'to_lower must be True or False (bool): {type(self.to_lower)}')

        # Check fail_none
        if self.fail_none not in {True, False}:
            raise ValueError(f'fail_none must be True or False (bool): {type(self.fail_none)}')

    def get_value(self, val):
        """Check and get value.

        :param val: Value to check.

        :returns: Value after checking and type conversion.

        :raises ValueError: If the value fails validation.
        """
        # Check default.
        if val is None:
            if self.fail_none:
                raise ValueError(f'Missing value for parameter {self.name}: Receieved None')

            val = self.default

        if val is None:
            return val

        # Check and cast type
        if self.val_type == 'int':
            try:
                val = int(val)
            except ValueError as e:
                raise ValueError(f'Failed casting {self.name} to int: {str(val)}') from e

        elif self.val_type == 'float':
            try:
                val = float(val)
            except ValueError as e:
                raise ValueError(f'Failed casting {self.name} to float: {str(val)}') from e

        elif self.val_type == 'bool':
            bool_val = as_bool(val, fail_to_none=True)
            val = bool_val

            if val is None:
                raise ValueError(f'Failed casting {self.name} to bool: {str(val)}')

        elif self.val_type == 'str':
            val = str(val)

        else:
            raise ValueError(f'Unrecognized parameter type (PROGRAM BUG) for {self.name}: {self.val_type}')

        # Convert to lower case
        if self.to_lower:
            if not self.val_type == 'str':
                raise ValueError(f'Cannot specify `to_lower=True` for non-string type {self.name}: {self.val_type}')

            val = val.lower()

        # Check allowed values
        if self.allowed is not None and val not in self.allowed:
            raise ValueError(f'Illegal value for {self.name}: {val} (allowed values: {self.allowed})')

        # Enforce min/max
        if self.min is not None:
            if isinstance(self.min, tuple):
                min_val, min_inclusive = self.min
            else:
                min_val, min_inclusive = self.min, True

            if val < min_val or (val == min_val and not min_inclusive):
                raise ValueError(
                    f'Illegal range for {self.name}: '
                    f'Minimum allowed value is {min_val} ({"inclusive" if min_inclusive else "exclusive"})'
                )

        if self.max is not None:
            if isinstance(self.max, tuple):
                max_val, max_inclusive = self.max
            else:
                max_val, max_inclusive = self.max, True

            if val > max_val or (val == max_val and not max_inclusive):
                raise ValueError(
                    f'Illegal range for {self.name}: '
                    f'Maximum allowed value is {max_val} ({"inclusive" if max_inclusive else "exclusive"})'
                )

        # Done converting and checking
        return val


_CONFIG_PARAM_LIST: list[_ConfigParamElement] = [

    # Alignments
    _ConfigParamElement(
        'aligner', 'str', allowed={'minimap2', 'lra'}, default='minimap2', is_null=True,
        description='Alignment program to use.'
    ),
    _ConfigParamElement(
        'align_params', 'str', is_null=True,
        description='Parameters for the aligner. Default depends on aligner (minimap2: "-x asm20"). '
                    'Keyword "pav2" reverts to legacy parameters used by PAV versions 1 & 2.'
    ),
    _ConfigParamElement(
        'lc_model', 'str', default='default',
        description='Low-confidence (LC) alignment prediction model. May be the name of a model packaged with'
                    'PAV or a path to a custom model. See "files/lcmodel/LC_MODEL.md for more information." in'
                    'the PAV distribution for more information',
        advanced=True
    ),
    _ConfigParamElement(
        'align_score_model', 'str', DEFAULT_ALIGN_SCORE_MODEL,
        description='Default alignment score model as a string argument to pav.align.score.get_score_model(). '
                    'These parameters are also used for scoring large variants.',
        advanced=True
    ),
    _ConfigParamElement(
        'redundant_callset', 'bool', False,
        description='Per haplotype assembly, callset is nonredundant per assembled sequence instead of globally '
                    'across all assembly sequences. Allows for multiple representations of the same locus '
                    'assembled in different sequences. May be useful for somatic variation, but requires more '
                    'significant downstream work, but will increase false-positive calls and requires more '
                    'downstream processind and QC to obtain a good-quality callset.',
        advanced=True
    ),
    _ConfigParamElement(
        'align_trim_max_depth', 'int', const.DEFAULT_ALIGN_TRIM_MAX_DEPTH, min=1,
        description='When trimming alignment records, filter out records where a proportion of the alignment '
                    'record is in regions with this depth or greater (see "align_trim_max_depth_prop").',
        advanced=True
    ),
    _ConfigParamElement(
        'align_trim_max_depth_prop', 'float', 0.8, min=0.0, max=1.0,
        description='When trimming alignment records, filter out records where this proportion of the '
                    'alignment record is in regions with depth greater than "align_trim_max_depth").',
        advanced=True
    ),

    # Variant calling
    _ConfigParamElement(
        'min_anchor_score', 'str', const.DEFAULT_MIN_ANCHOR_SCORE,
        description='Minimum score of an aligned segment to allow it to be used as an anchor. This value may '
                    'be the absolute score value or a relative value adjusted for the score of a perfectly '
                    'aligned segment of some length (e.g. "1000bp" would be the score of 1000 aligned bases '
                    'with no gaps or mismatches, i.e. 2000 with default alignment parameters with match=2). '
                    'Any alignment record with a score of at least this value may be used as an anchor for '
                    'alignment-truncating variants.'
    ),
    _ConfigParamElement(
        'lg_off_gap_mult', 'float', const.DEFAULT_LG_OFF_GAP_MULT,
        min=(1.0, False),
        description='Large variants are penalized for gaps inconsistent with their variant type, e.g. a '
                    'reference gap (del) at an insertion site. For these off-gaps, multiply the gap score'
                    'by this factor (see parameter "align_score_model" for gap scores).',
        advanced=True
    ),
    _ConfigParamElement(
        'lg_gap_scale', 'float', const.DEFAULT_LG_GAP_SCALE,
        min=(0.0, False),
        description='Alignment anchoring candidate SVs are ignored if the penalty of the gap between two '
                    'candidate anchor alignments (reference gap) is greater than the alignment score of '
                    'either anchor. The gap score between anchors is multiplied by this value before it is '
                    'compared to the anchor scores. A value of less than 1.0 reduces the gap penalty (i.e. '
                    'allows smaller alignments to anchor larger variants), and a value greater than 1.0 '
                    'increases the gap penalty (i.e. variant require more substantial anchoring alignments. '
                    'See parameter "align_score_model" for how gap and anchor alignments are score.',
        advanced=True
    ),
    _ConfigParamElement(
        'lg_smooth_segments', 'float', const.DEFAULT_LG_SMOOTH_SEGMENTS,
        min=(0.0, True),
        description='For complex variant calls, smooth aligned segments concatenating adjacent segments if '
                    'they are this proportion or smaller than the total SV length. The full structure of SVs '
                    'is accessible in the variant call, but reference and query traces are reported with '
                    'smoothing applied.',
        advanced=True
    ),
    _ConfigParamElement(
        'lg_cpx_min_aligned_prop', 'float', default=0.8,
        min=(0.0, True), max=(1.0, True),
        description='For complex variant calls, require this proportion of the total SV length to be aligned '
                    'to the reference sequence.',
        advanced=True
    ),
    _ConfigParamElement(
        'vcf_haplotypes', 'str', None,
        description='A comma-separated list of haplotype names to include in the VCF where genotypes are given in this'
                    'order. If not defined, then all haplotypes are written in the order they are defined.'
    ),

    # Inversion site flagging from variant call clusters
    _ConfigParamElement(
        'inv_sig_cluster_flank', 'int', 100,
        description='Cluster SNV & indel variants within this many bases upstream or downstream of variant midpoints.'
    ),
    _ConfigParamElement(
        'inv_sig_cluster_win_min', 'int', 500,
        description='Minimum size of SNV & indel cluster windows.'
    ),
    _ConfigParamElement(
        'inv_sig_cluster_snv_min', 'int', 20,
        description='Minimum depth of SNVs in a cluster window.'
    ),
    _ConfigParamElement(
        'inv_sig_cluster_indel_min', 'int', 10,
        description='Minimum depth of indels in a cluster window.'
    ),
    _ConfigParamElement(
        'inv_sig_cluster_varlen_min', 'int', 1,
        description='Discard indels less than this size.'
    ),
    _ConfigParamElement(
        'inv_sig_insdel_offset_prop', 'float', 2.0,
        description='Offset proportion (offset_prop_max) when intersecting INS and DEL variants of similar size.'
    ),
    _ConfigParamElement(
        'inv_sig_insdel_varlen_ro', 'float', 0.8,
        description='Variant length overlap proportion (size_ro_min) when intersecting INS and DEL variants of similar '
                    'size.'
    ),
    _ConfigParamElement(
        'inv_sig_merge_flank', 'int', 1000,
        description='Merge windows within this many bp.'
    ),
    _ConfigParamElement('inv_max_overlap', 'float', 0.2,
                        min=0.0, max=1.0,
                        description='Maximum allowed reciprocal overlap between inversions in the same haplotype.'),

    # Inversions
    _ConfigParamElement(
        'inv_min', 'int', 0, min=0, description='Minimum inversion size.'
    ),
    _ConfigParamElement(
        'inv_max', 'int', 0, min=0, description='Maximum inversion size. Unlimited inversion size if value is 0.'
    ),

    _ConfigParamElement(
        'inv_region_limit', 'int', const.INV_REGION_LIMIT,
        description='maximum region size when searching for inversions. Value 0 ignores limits and allows regions to '
                    'be any size.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_min_expand', 'int', const.INV_MIN_EXPAND_COUNT,
        description='The default number of region expansions to try (including the initial expansion) and '
                    'finding only fwd k-mer states after smoothing before giving up on the region.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_init_expand', 'int', const.INV_INIT_EXPAND,
        description='Expand the flagged region by this (bp) before starting.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_min_kmers', 'int', const.INV_MIN_KMERS,
        description='Minimum number of k-mers with a distinct state (sum of FWD, FWDREV, and REV). Stop if the '
                    'number of k-mers is less after filtering uninformative and high-count k-mers.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_max_ref_kmer_count', 'int', const.INV_MAX_REF_KMER_COUNT,
        description='If canonical reference k-mers have a higher count than this, they are discarded.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_repeat_match_prop', 'float', const.INV_REPEAT_MATCH_PROP,
        description='When scoring INV structures, give a bonus to inverted repeats that are similar in size '
                    'scaled by this factor.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_min_kmer_run', 'int', const.INV_MIN_INV_KMER_RUN,
        description='Minimum continuous run of strictly inverted k-mers.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_min_qry_ref_prop', 'float', const.INV_MIN_QRY_REF_PROP,
        description='Minimum query and reference region size proportion.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_k_size', 'int', const.INV_K_SIZE, description='K-mer size.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_kde_bandwidth', 'float', const.INV_KDE_BANDWIDTH,
        description='Convolution KDE bandwidth.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_kde_trunc_z', 'float', const.INV_KDE_TRUNC_Z,
        description='Convolution KDE truncated normal Z-score based on a standard normal (N(0,1)) distribution.',
        advanced=True
    ),
    _ConfigParamElement(
        'inv_kde_func', 'str', const.INV_KDE_FUNC, allowed={'auto', 'fft', 'conv'}, to_lower=True,
        description='Convolution method. "fft" uses a Fast-Fourier Transform, "conv" is a standard truncated '
                    'normal distribution. "auto" defaults to "fft" if scipy.signal is available and "conv" '
                    'otherwise.',
        advanced=True
    ),

    # Troubleshooting and verbosity
    _ConfigParamElement(
        'verbose', 'bool', default=False,
        description='Verbose output.'
    ),
    _ConfigParamElement(
        'debug', 'bool', default=False, advanced=True,
        description='Extra debugging checks. This option may slow down the pipeline significantly and should be used '
                    'for testing only. Do not enable in production.'
    )
]
"""List of known configuration parameters, default values, documentation, and validation rules."""


class _ConfigParamMap:
    """Dictionary of known configuration parameters."""

    def __init__(self) -> None:
        """Initialize the dictionary."""
        self._map = dict()
        for param in _CONFIG_PARAM_LIST:
            self._map[param.name] = param

    def __getitem__(self, key: str) -> _ConfigParamElement:
        """Get a parameter by name."""
        return self._map[key]

    def keys(self):
        """Get parameter keys."""
        return self._map.keys()


CONFIG_PARAM_DICT: _ConfigParamMap = _ConfigParamMap()
"""Dictionary of known configuration parameters.

Each parameter has default values, documentation, and validation rules keyed by parameter name.
"""


def format_config_md(
        out_file: TextIO = sys.stdout,
        width: int = 80,
        advanced: bool = True
):
    """Write markdown-formatted help for configuration options.

    :param out_file: Output file.
    :param width: Line-wrap length.
    :param advanced: Include advanced options.
    """
    for param in _CONFIG_PARAM_LIST:

        if not advanced and param.advanced:
            continue

        first_line = f'* {param.name} [{param.val_type}'

        if param.default is not None:
            if param.val_type == 'str':
                first_line += f', "{param.default}"'
            else:
                first_line += f', {param.default}'

        if param.min is not None or param.max is not None:
            if param.min is not None:
                range = ('[' if isinstance(param.min, tuple) and param.min[1] else '(') + str(param.min) + ':'
            else:
                range = '(-inf : '

            if param.max is not None:
                range += str(param.max) + (']' if isinstance(param.max, tuple) and param.max[1] else ')')
            else:
                range += 'inf)'

            first_line += f', {range}'

        if param.allowed is not None:
            first_line += f', {param.allowed}'

        first_line += ']: '

        out_file.write(
            '\n'.join(textwrap.wrap(param.description, initial_indent=first_line, subsequent_indent='  ', width=width))
        )

        out_file.write('\n')


def get_align_params(aligner, align_params):
    """Get alignment parameters.

    :returns: A string of parameters for the aligner. Will pull from default values if not overridden.
    """
    if align_params is None:
        return DEFAULT_ALIGNER_PARAMS.get(aligner, None)

    named_key = align_params.strip().lower()

    if named_key in NAMED_ALIGNER_PARAMS.keys():
        if aligner not in NAMED_ALIGNER_PARAMS[align_params.lower()]:
            raise RuntimeError(
                f'Named alignment parameters are not defined for this aligner: {align_params}, aligner={aligner}'
            )

        return NAMED_ALIGNER_PARAMS[align_params.lower()][aligner]

    return align_params
