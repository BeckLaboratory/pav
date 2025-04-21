"""
Manage pipeline configurations
"""

import builtins
import pandas as pd
import sys
import textwrap

from . import align
from . import const
from . import util as pav_util

DEFAULT_ALIGNER = 'minimap2'

DEFAULT_ALIGNER_PARAMS = {
    'minimap2': '-x asm5',
    'lra': '',
}

NAMED_ALIGNER_PARAMS = {
    'pav2': {
        'minimap2': '-x asm20 -m 10000 -z 10000,50 -r 50000 --end-bonus=100 -O 5,56 -E 4,1 -B 5',
        'lra': ''
    }
}

KNOWN_ALIGNERS = ['minimap2', 'lra']


#
# Configuration parameters
#

class ConfigParams(object):
    aligner: str
    aligner_params: str
    lc_model: str
    align_score_model: str
    redundant_callset: bool
    no_link_qry: bool
    align_agg_min_score: float
    align_agg_noncolinear_penalty: bool
    merge_partitions: int
    cigar_partitions: int
    query_filter: str
    min_anchor_score: str  # Formatted string may contain scaling factors
    lg_off_gap_mult: float
    lg_gap_scale: float
    lg_smooth_segments: float
    lg_cpx_min_aligned_prop: float
    inv_sig_cluster_win: int
    inv_sig_cluster_win_min: int
    inv_sig_cluster_snv_min: int
    inv_sig_cluster_indel_min: int
    inv_sig_insdel_cluster_flank: int
    inv_sig_insdel_merge_flank: int
    inv_sig_cluster_svlen_min: int
    inv_sig_merge_flank: int
    inv_sig_part_count: int
    inv_sig_filter: str
    inv_max_overlap: float
    inv_min: int
    inv_max: int
    inv_region_limit: int
    inv_min_expand: int
    inv_init_expand: int
    inv_min_kmers: int
    inv_max_ref_kmer_count: int
    inv_repeat_match_prop: float
    inv_min_inv_kmer_run: int
    inv_min_qry_ref_prop: float
    inv_k_size: int
    inv_kde_bandwidth: float
    inv_kde_trunc_z: float
    inv_kde_func: str
    verbose: bool

    def __init__(self,
                 asm_name: str=None,
                 config: dict=None,
                 asm_table: pd.DataFrame=None,
                 verbose: bool=None
                 ):

        self._asm_name = asm_name
        self._config = config if config is not None else dict()
        self._asm_table = asm_table

        self._set_config_override_dict()

        if verbose is None:
            self.verbose = CONFIG_PARAM_DICT['verbose'].get_value(
                self._config_override.get('verbose', self._config.get('verbose', None))
            )
        else:
            self.verbose = verbose

    def __getattr__(self, key):
        """
        Get a configuration parameter, set if needed. Report
        """

        if key not in CONFIG_PARAM_DICT.keys():
            raise KeyError(f'Unknown configuration parameter "{key}"')

        if key == 'align_params':
            setattr(
                self,
                key,
                get_align_params(self.aligner,
                    CONFIG_PARAM_DICT[key].get_value(
                        self._config_override.get(key, self._config.get(key, None))
                    )
                )
            )

        else:
            setattr(
                self,
                key,
                CONFIG_PARAM_DICT[key].get_value(
                    self._config_override.get(key, self._config.get(key, None))
                )
            )

        val = getattr(self, key)

        if self.verbose:
            print(f'Config: {key} = {val} [{str(self)}]')

        return val

    def _set_config_override_dict(self):
        """
        Get a dictionary of overridden parameters using the CONFIG column of the assembly table.

        :param config_string: Config override string (e.g. attr1=val1;attr2=val2). Must be colon separated and each
            element must have an equal sign. Whitespace around semi-colons and equal signs is ignored.

        :return: Dict of overridden parameters or an empty dict if no parameters were overridden.
        """

        # Init config override
        self._config_override = dict()

        # Get override config string
        if self._asm_table is None or self._asm_name is None or self._asm_name not in self._asm_table.index:
            return

        config_string = self._asm_table.loc[self._asm_name]['CONFIG']

        if config_string is None or pd.isnull(config_string) or not config_string.strip():
            return

        config_string = config_string.strip()

        # Process each config directive
        tok_list = config_string.split(';')

        for tok in tok_list:

            # Check tok
            tok = tok.strip()

            if not tok:
                continue

            if '=' not in tok:
                raise RuntimeError('Cannot get assembly config: Missing "=" in CONFIG token {}: {}'.format(tok, config_string))

            # Get attribute and value
            key, val = tok.split('=', 1)

            key = key.strip()
            val = val.strip()

            if not key:
                raise RuntimeError('Cannot get assembly config: Missing key (key=value) in CONFIG token {}: {}'.format(tok, config_string))

            if not val:
                raise RuntimeError('Cannot get assembly config: Missing value (key=value) in CONFIG token {}: {}'.format(tok, config_string))

            # Set value
            self._config_override[key] = val

    def get_aligner_input(self):
        """
        :return: A list of input files for this aligner and sample.
        """

        # Check parameters
        aligner = self.aligner

        # Return list of input file (FASTA file is first)
        if aligner == 'minimap2':
            return [
                'data/query/{asm_name}/query_{hap}.fa.gz',
                'data/query/{asm_name}/query_{hap}.fa.gz.fai',
                'data/query/{asm_name}/query_{hap}.fa.gz.gzi'
            ]

        if aligner == 'lra':
            return [
                'temp/{asm_name}/align/query/query_{hap}.fa',
                'data/ref/ref.fa.gz.gli',
                'data/ref/ref.fa.gz.mms'
            ]

        raise RuntimeError(f'Unknown aligner: {aligner}')

    def __repr__(self):
        return f'ConfigParams(asm_name={self._asm_name}, config={'CONFIG' if self._config is not None else 'None'}, asm_table={'ASM_TABLE' if self._asm_table is not None else 'None'})'


class ConfigParamElement(object):
    name: str
    val_type: str
    default: any
    min: any
    max: any
    allowed: set[any]
    to_lower: bool
    fail_none: bool
    description: str
    is_null: bool
    advanced: bool
    """
    A configuration parameter object.

    Minimum and maximum values can be either a single value to check or a tuple of (value, inclusive/exclusive). If
    it is a single value, then the min/max is inclusive. If it is a tuple, then min/max is inclusive if the second
    element of the tuple is `True` and exclusive if it is `False`.

    Fields:
    * name: Parameter name.
    * val_type: Type of parameter as a string.
    * default: Default value.
    * min: Minimum value if not `None`.
    * max: Maximum value if not `None`.
    * allowed: Set of allowed values if not `None`.
    * to_lower: String value is converted to lower case if `True`. Only valid if `val_type` is `str`.
    * fail_none: If `True`, fail if a parameter value is `None`, otherwise, return the default value.
    * description: Description of the parameter.
    * is_null: `True` if the parameter is included for documentation purposes, but parameter processing is handled
        outside this class. Alignment parameters must be adjusted for the aligner, and so it is not processed here,
        however, the alignment ConfigParam objects are included to simplify parameter documentation.
    * advanced: `True` if the parameter is an advanced option and should not be shown in brief documentation.
    """

    def __init__(self,
                 name, val_type,
                 default=None,
                 min=None, max=None, allowed=None,
                 to_lower=False, fail_none=False,
                 description=None,
                 is_null=False,
                 advanced=False
                 ):
        self.name = name
        self.val_type = val_type
        self.default = default
        self.min = min
        self.max = max
        self.allowed = allowed
        self.to_lower = to_lower
        self.fail_none = fail_none
        self.description = description
        self.is_null = is_null
        self.advanced = advanced

        # Check name
        if self.name is None or not isinstance(self.name, str) or not self.val_type.strip():
            raise RuntimeError('name is missing or empty')

        self.name = self.name.strip().lower()

        # Check type
        if self.val_type is None or not isinstance(self.val_type, str) or not self.val_type.strip():
            raise RuntimeError('Type is missing or empty')

        self.val_type = self.val_type.strip().lower()

        if self.val_type not in {'int', 'float', 'bool', 'str'}:
            raise RuntimeError(f'Unrecognized parameter type: {self.val_type}')

        # Check remaining parameters
        if allowed is not None and not isinstance(allowed, set):
            raise RuntimeError(f'Allowed values must be a set: {type(allowed)}')

        if to_lower not in {True, False}:
            raise RuntimeError(f'to_lower must be True or False (bool): {type(to_lower)}')

        if fail_none not in {True, False}:
            raise RuntimeError(f'fail_none must be True or False (bool): {type(fail_none)}')

    def get_value(self, val):
        """
        Check and get value.

        :param val: Value to check.

        :return: Value after checking and type conversion.
        """

        # Check default.
        if val is None:
            if self.fail_none:
                raise RuntimeError(f'Missing value for parameter {self.name}: Receieved None')

            val = self.default

        if val is None:
            return val

        # Check and cast type
        if self.val_type == 'int':
            try:
                val = int(val)
            except ValueError:
                raise RuntimeError(f'Failed casting {self.name} to int: {str(val)}')

        elif self.val_type == 'float':
            try:
                val = float(val)
            except ValueError:
                raise RuntimeError(f'Failed casting {self.name} to float: {str(val)}')

        elif self.val_type == 'bool':
            bool_val = pav_util.as_bool(val, fail_to_none=True)
            val = bool_val

            if val is None:
                raise RuntimeError(f'Failed casting {self.name} to bool: {str(val)}')

        elif self.val_type == 'str':
            val = str(val)

        else:
            raise RuntimeError(f'Unrecognized parameter type (PROGRAM BUG) for {self.name}: {self.val_type}')

        # Convert to lower case
        if self.to_lower:
            if not self.val_type == 'str':
                raise RuntimeError(f'Cannot specify `to_lower=True` for non-string type {self.name}: {self.val_type}')

            val = val.lower()

        # Check allowed values
        if self.allowed is not None and val not in self.allowed:
            raise RuntimeError(f'Illegal value for {self.name}: {val} (allowed values: {self.allowed})')

        # Enforce min/max
        if self.min is not None:
            if isinstance(self.min, tuple):
                min_val, min_inclusive = self.min
            else:
                min_val, min_inclusive = self.min, True

            if val < min_val or (val == min_val and not min_inclusive):
                raise RuntimeError(f'Illegal range for {self.name}: Minimum allowed value is {min_val} ({"inclusive" if min_inclusive else "exclusive"})')

        if self.max is not None:
            if isinstance(self.max, tuple):
                max_val, max_inclusive = self.max
            else:
                max_val, max_inclusive = self.max, True

            if val > max_val or (val == max_val and not max_inclusive):
                raise RuntimeError(f'Illegal range for {self.name}: Maximum allowed value is {max_val} ({"inclusive" if max_inclusive else "exclusive"})')

        # Done converting and checking
        return val


CONFIG_PARAM_LIST = [

    # Alignments
    ConfigParamElement('aligner', 'str', allowed={'minimap2', 'lra'}, default='minimap2', is_null=True,
                description='Alignment program to use.'),
    ConfigParamElement('align_params', 'str', is_null=True,
                description='Parameters for the aligner. Default depends on aligner (minimap2: "-x asm5", lra: ""). '
                            'Keyword "pav2" reverts to legacy parameters used by PAV versions 1 & 2.'),
    ConfigParamElement('lc_model', 'str', default='default',
                description='Low-confidence (LC) alignment prediction model. May be the name of a model packaged with'
                            'PAV or a path to a custom model. See "files/lcmodel/LC_MODEL.md for more information." in'
                            'the PAV distribution for more information',
                advanced=True),
    ConfigParamElement('align_score_model', 'str', align.score.DEFAULT_ALIGN_SCORE_MODEL,
                description='Default alignment score model as a string argument to pavlib.align.score.get_score_model(). '
                            'These parameters are also used for scoring large variants.',
                advanced=True),
    ConfigParamElement('redundant_callset', 'bool', False,
                description='Per haplotype assembly, callset is nonredundant per assembled sequence instead of globally '
                            'across all assembly sequences. Allows for multiple representations of the same locus '
                            'assembled in different sequences. May be useful for somatic variation, but requires more '
                            'significant downstream work, but will increase false-positive calls and requires more '
                            'downstream processind and QC to obtain a good-quality callset.',
                advanced=True),
    ConfigParamElement('no_link_qry', 'bool', False,
                description='If set, always copy the input query (assembly) FASTA file into the run directory and '
                            're-write. By default (False), the input query file is only stored in the run directory if '
                            'it is not a single compressed FASTA file (e.g. plain text FASTA, multiple FASTAs, or GFA '
                            'input). This option is helpful if there are input FASTA irregularities that cause '
                            'downstream steps to fail, but it requires time and space to reformat and store input '
                            'files. Not recommended for most use cases.'),
    ConfigParamElement('align_agg_min_score', 'float', -10000.0, max=0.0,
                description='Aggregate alignment records that are adjacent if the sum of the gap scores (reference gap '
                            'and query gap) is not less than this value (gap scores are negative). Two or more '
                            'alignment records are merged into one. A value of 0.0 disables alignment aggregation. '
                            'scores are computed based on "align_score_model" values.'),
    ConfigParamElement('align_agg_noncolinear_penalty', 'bool', True,
                description='Penalize alignment records that are adjacent by the difference in lengths of the '
                            'reference and query gaps (i.e. add "gap(abs(ref_gap - qry_gap))" to the gap score when '
                            'choosing to aggregate or not (the total sum must be greater than align_agg_min_score).',
                advanced=True),
    ConfigParamElement('align_trim_max_depth', 'int', 20, min=1,
                description='When trimming alignment records, filter out records where a proportion of the alignment '
                            'record is in regions with this depth or greater (see "align_trim_max_depth_prop").',
                advanced=True),
    ConfigParamElement('align_trim_max_depth_prop', 'float', 0.8, min=0.0, max=1.0,
                description='When trimming alignment records, filter out records where this proportion of the '
                            'alignment record is in regions with depth greater than "align_trim_max_depth").',
                advanced=True),


    # Variant calling
    ConfigParamElement('merge_partitions', 'int', 20, min=1,
                description='Split variants into this many partitions to merge.',
                advanced=True),
    ConfigParamElement('cigar_partitions', 'int', 10, min=1,
                description='For intra-alignment (not alignment truncating), split chromosomes into this many '
                            'partitions and search for INS/DEL/SNV inside alignment records for each partition.',
                advanced=True),
    ConfigParamElement('query_filter', 'str', None,
                description='Query filter BED file. May be multiple file names separated by semicolons (";"). Each BED '
                            'file contains regions in query-coordinates (assembled sequence) matching sequences in the '
                            'input FASTA file. Any variants intersecting these loci are dropped from the callset. May '
                            'be used to  apply quality filtering for known mis-assembled loci.'),
    ConfigParamElement('min_anchor_score', 'str', const.DEFAULT_MIN_ANCHOR_SCORE,
                description='Minimum score of an aligned segment to allow it to be used as an anchor. This value may '
                            'be the absolute score value or a relative value adjusted for the score of a perfectly '
                            'aligned segment of some length (e.g. "1000bp" would be the score of 1000 aligned bases '
                            'with no gaps or mismatches, i.e. 2000 with default alignment parameters with match=2). '
                            'Any alignment record with a score of at least this value may be used as an anchor for '
                            'alignment-truncating variants.'),
    ConfigParamElement('lg_off_gap_mult', 'float', const.DEFAULT_LG_OFF_GAP_MULT,
                min=(1.0, False),
                description='Large variants are penalized for gaps inconsistent with their variant type, e.g. a '
                            'reference gap (del) at an insertion site. For these off-gaps, multiply the gap score'
                            'by this factor (see parameter "align_score_model" for gap scores).',
                advanced=True
    ),
    ConfigParamElement('lg_gap_scale', 'float', const.DEFAULT_LG_GAP_SCALE,
                min=(0.0, False),
                description='Alignment anchoring candidate SVs are ignored if the penalty of the gap between two '
                            'candidate anchor alignments (reference gap) is greater than the alignment score of '
                            'either anchor. The gap score between anchors is multiplied by this value before it is '
                            'compared to the anchor scores. A value of less than 1.0 reduces the gap penalty (i.e. '
                            'allows smaller alignments to anchor larger variantns), and a value greater than 1.0 '
                            'increases the gap penalty (i.e. variant require more substantial anchoring alignments. '
                            'See parameter "align_score_model" for how gap and anchor alignments are score.',
                advanced=True
    ),
    ConfigParamElement('lg_smooth_segments', 'float', const.DEFAULT_LG_SMOOTH_SEGMENTS,
                min=(0.0, True),
                description='For complex variant calls, smooth aligned segments concatenating adjacent segments if '
                            'they are this proportion or smaller than the total SV length. The full structure of SVs '
                            'is accessible in the variant call, but reference and query traces are reported with '
                            'smoothing applied.',
                advanced=True
    ),
    ConfigParamElement('lg_cpx_min_aligned_prop', 'float', default=0.8,
                min=(0.0, True), max=(1.0, True),
                description='For complex variant calls, require this proportion of the total SV length to be aligned '
                            'to the reference sequence.',
                advanced=True),

    # Inversion site flagging from variant call clusters
    ConfigParamElement('inv_sig_cluster_win', 'int', 200,
                description='Cluster variants within this many bases.'),
    ConfigParamElement('inv_sig_cluster_win_min', 'int', 500,
                description='Window must reach this size.'),
    ConfigParamElement('inv_sig_cluster_snv_min', 'int', 20,
                description='Minimum number if SNVs in window.'),
    ConfigParamElement('inv_sig_cluster_indel_min', 'int', 10,
                description='Minimum number of indels in window.'),
    ConfigParamElement('inv_sig_insdel_cluster_flank', 'int', 2,
                description='For each insertion, multiply SVLEN by this to get the distance to the nearest deletion it may intersect.'),
    ConfigParamElement('inv_sig_insdel_merge_flank', 'int', 2000,
                description='Merge clusters within this distance (bp).'),
    ConfigParamElement('inv_sig_cluster_svlen_min', 'int', 4,
                description='Discard indels less than this size.'),
    ConfigParamElement('inv_sig_merge_flank', 'int', 500,
                description='Merge windows within this many bp.'),
    ConfigParamElement('inv_sig_part_count', 'int', 10,
                description='Partition signature regions into this many partitions for the caller. Marked here so that this file can be cross-referenced with the inversion caller log.'),
    ConfigParamElement('inv_sig_filter', 'str', 'svindel',
                description='Filter flagged regions.'),
    ConfigParamElement('inv_max_overlap', 'float', 0.2,
                min=0.0, max=1.0,
                description='Maximum allowed reciprocal overlap between inversions in the same haplotype.'),

    # Inversions
    ConfigParamElement('inv_min', 'int', 0, min=0, description='Minimum inversion size.'),
    ConfigParamElement('inv_max', 'int', 0, min=0, description='Maximum inversion size. Unlimited inversion size if value is 0.'),
    # ConfigParamElement('inv_inner', 'str', 'core',
    #             allowed={'core', 'none', 'full'}, to_lower=True,
    #             description='Filter variants inside the inverted core (no flanking repeats) inversions if "core", '
    #                 'full inversion including repeats if "full", and do not filter if "none".',
    #             advanced=True),

    ConfigParamElement('inv_region_limit', 'int', const.INV_REGION_LIMIT,
                description='maximum region size when searching for inversions. Value 0 ignores limits and allows regions to be any size.',
                advanced=True),
    ConfigParamElement('inv_min_expand', 'int', const.INV_MIN_EXPAND_COUNT,
                description='The default number of region expansions to try (including the initial expansion) and '
                            'finding only fwd k-mer states after smoothing before giving up on the region.',
                advanced=True),
    ConfigParamElement('inv_init_expand', 'int', const.INV_INIT_EXPAND,
                description='Expand the flagged region by this (bp) before starting.',
                advanced=True),
    ConfigParamElement('inv_min_kmers', 'int', const.INV_MIN_KMERS,
                description='Minimum number of k-mers with a distinct state (sum of FWD, FWDREV, and REV). Stop if the '
                            'number of k-mers is less after filtering uninformative and high-count k-mers.',
                advanced=True),
    ConfigParamElement('inv_max_ref_kmer_count', 'int', const.INV_MAX_REF_KMER_COUNT,
                description='If canonical reference k-mers have a higher count than this, they are discarded.',
                advanced=True),
    ConfigParamElement('inv_repeat_match_prop', 'float', const.INV_REPEAT_MATCH_PROP,
                description='When scoring INV structures, give a bonus to inverted repeats that are similar in size '
                            'scaled by this factor.',
                advanced=True),
    ConfigParamElement('inv_min_inv_kmer_run', 'int', const.INV_MIN_INV_KMER_RUN,
                description='Minimum continuous run of strictly inverted k-mers.',
                advanced=True),
    ConfigParamElement('inv_min_qry_ref_prop', 'float', const.INV_MIN_QRY_REF_PROP,
                description='Minimum query and reference region size proportion.',
                advanced=True),

    ConfigParamElement('inv_k_size', 'int', const.INV_K_SIZE, description='K-mer size.', advanced=True),

    ConfigParamElement('inv_kde_bandwidth', 'float', const.INV_KDE_BANDWIDTH,
                description='Convolution KDE bandwidth.',
                advanced=True),
    ConfigParamElement('inv_kde_trunc_z', 'float', const.INV_KDE_TRUNC_Z,
                description='Convolution KDE truncated normal Z-score based on a standard normal (N(0,1)) distribution.',
                advanced=True),
    ConfigParamElement('inv_kde_func', 'str', const.INV_KDE_FUNC, allowed={'auto', 'fft', 'conv'}, to_lower=True,
                description='Convolution method. "fft" uses a Fast-Fourier Transform, "conv" is a standard truncated '
                            'normal distribution. "auto" defaults to "fft" if scipy.signal is available and "conv" '
                            'otherwise.',
                advanced=True),

    # Misc
    ConfigParamElement('verbose', 'bool', default=False,
                       description='Verbose output.'),
    ConfigParamElement('debug', 'bool', default=False,
                       description='Extra debugging checks.')
]

CONFIG_PARAM_DICT = {
    param.name: param for param in CONFIG_PARAM_LIST
}


def format_config_md(out_file=sys.stdout, width=80, advanced=True):
    """
    Write markdown-formatted help for configuration options.

    :param out_file: Output file.
    :param width: Line-wrap length.
    :param advanced: Include advanced options.
    """

    for param in CONFIG_PARAM_LIST:

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


#
# Configuration retrieval
#




# def get_config_with_override(config, override_config):
#     """
#     Get a config dict with values replaced by overridden values. The dict in parameter `config` is copied if it is
#     modified. The original (unmodified) config or a modified copy is returned.
#
#     :param config: Existing config. Original object will not be modified.
#     :param override_config: A defined set of values that will override entries in `config`.
#
#     :return: A config object.
#     """
#
#     if override_config is None:
#         return config
#
#     if config is None:
#         config = dict()
#
#     config = config.copy()
#
#     for key, val in override_config.items():
#         if key in {'reference'}:
#             raise RuntimeError('The reference configuration parameter cannot be overridden or defined per sample.')
#
#         config[key] = val
#
#     return config


# def get_override_config(asm_name, config, asm_table):
#     """
#     Get a config dict with values replaced by overridden values. The dict in parameter `config` is copied if it is
#     modified. The original (unmodified) config or a modified copy is returned.
#
#     :param asm_name: Name of the assembly.
#     :param config: Existing config. Original object will not be modified.
#     :param asm_table: Assembly table (DataFrame).
#
#     :return: A config object.
#     """
#
#     if asm_name is None:
#         raise RuntimeError('Cannot get config overide for assembly: None')
#
#     if asm_table is None:
#         raise RuntimeError('Cannot get config overide for assembly table: None')
#
#     # Get table entry
#     if asm_name not in asm_table.index:
#         return config
#
#     asm_table_entry = asm_table.loc[asm_name]
#
#     if 'CONFIG' not in asm_table_entry:
#         return config
#
#     return get_config_with_override(config, get_config_override_dict(asm_table_entry['CONFIG']))


# def get_config(key, asm_name=None, config=None, asm_table=None):
#     """
#     Get a config object that might be modified by CONFIG parameters in the assembly table.
#
#     If "key" is None, the full config dictionary is returned. If "key" is defined, then the value of config for
#     that key is returned with an optional default value.
#
#     :param key: Key of the value to get from config.
#     :param asm_name: Rule wildcards with assembly na
#     :param config: Global pipeline config (dict).
#     :param asm_table: Assembly table (DataFrame).
#
#     :return: Validated value of a configuration parameter cast to the correct type for that value.
#     """
#
#     if asm_name is not None:
#         if asm_table is None:
#             raise RuntimeError('Cannot get config for assembly "{wildcards.asm_name}": Assembly table is None')
#
#         local_config = get_override_config(asm_name, config, asm_table)
#
#     else:
#         local_config = config
#
#     if local_config is None:
#         local_config = dict()
#
#     if key is None:
#         raise RuntimeError(f'key is missing')
#
#     if key not in CONFIG_PARAM_DICT:
#         raise RuntimeError(f'Unknown parameter "{key}" for assembly "{asm_name}"')
#
#     return CONFIG_PARAM_DICT[key].get_value(local_config[key] if key in local_config else None)

def get_align_params(aligner, align_params):
    """
    Get alignment parameters.

    :return: A string of parameters for the aligner. Will pull from default values if not overridden by config.
    """

    if align_params is None:
        return DEFAULT_ALIGNER_PARAMS.get(aligner, None)

    named_key = align_params.strip().lower()

    if named_key in NAMED_ALIGNER_PARAMS.keys():
        if aligner not in NAMED_ALIGNER_PARAMS[align_params.lower()]:
            raise RuntimeError(f'Named alignment parameters are not defined for this aligner: {align_params}, aligner={aligner}')

        return NAMED_ALIGNER_PARAMS[align_params.lower()][aligner]

    return align_params

# class ConfigDict(dict):
#     def __getattr__(self, name):
#         if name in self.keys():
#             return self[name]
#
#         raise ValueError(f'Unknown parameter: "{name}"')


# def get_config_dict(asm_name=None, config=None, asm_table=None):
#     """
#     Get a dictionary of configuration values.
#
#     :param asm_name: Assembly name (None ignores overrides per sample in the sample table).
#     :param config: Pipeline config (None for defaults)
#     :param asm_table: Assembly table. May be None unless asm_name is not None.
#
#     :return: A dictionary of assembly parameters.
#     """
#
#     config_dict = ConfigDict()
#
#     for key in CONFIG_PARAM_DICT.keys():
#         config_dict[key] = get_config(key, asm_name, config, asm_table)
#
#     return config_dict

#
# Aligner parameters
#

