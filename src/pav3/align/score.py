"""Alignment routines."""

__all__ = [
    'AFFINE_SCORE_MATCH',
    'AFFINE_SCORE_MISMATCH',
    'AFFINE_SCORE_GAP',
    'AFFINE_SCORE_TS',
    'DEFAULT_ALIGN_SCORE_MODEL',
    'ScoreModel',
    'AffineScoreModel',
    'get_score_model',
    'get_affine_by_params',
]

from abc import ABC, abstractmethod
import re
from typing import Any, Optional

import numpy as np
import polars as pl

from agglovar.meta.decorators import immutable

from . import op

AFFINE_SCORE_MATCH: float = 2.0
"""Match score (minimap2 default)"""

AFFINE_SCORE_MISMATCH: float = 4.0
"""Mismatch score (minimap2 default)"""

AFFINE_SCORE_GAP: tuple[tuple[float, float], ...] = ((4.0, 2.0), (24.0, 1.0))
"""Multi-affine scores as a sequence of gap-open and gap-extend penalties (minimap2 defaults)."""

AFFINE_SCORE_TS = (1.0, 50)
"""Template switch score.

May be a single float (penalty of a template switch) or a tuple of a multiplier and a gap size
(e.g. "(1.0, 50)" sets the template switch score to the cost of a 50 bp gap).
"""

AFFINE_OFF_VARIANT_PENALTY = 2.0
"""Multiplier for off-variant penalties when scoring.

For example, the gap score of query (inserted) bases for a deletion would be multiplied by this
factor. Used for picking the right variant call type for a set of alignment features.
"""

DEFAULT_ALIGN_SCORE_MODEL = (
    f'affine::'
    f'match={AFFINE_SCORE_MATCH},'
    f'mismatch={AFFINE_SCORE_MISMATCH},'
    f'gap={";".join([f"{gap_open}:{gap_extend}" for gap_open, gap_extend in AFFINE_SCORE_GAP])},'
    f'ts={str(AFFINE_SCORE_TS).replace(',', ':')},'
    f'off_var={AFFINE_OFF_VARIANT_PENALTY}'
)
"""Default alignment score model specification."""


class ScoreModel(ABC):
    """Score model interface."""

    @abstractmethod
    def match(self, n: int = 1) -> float:
        """Score matching bases.

        :param n: Number of matching bases.

        :returns: Match score.
        """
        ...

    @abstractmethod
    def mismatch(self, n: int = 1) -> float:
        """Score mismatched bases.

        :param n: Number of mismatched bases.

        :returns: Mismatch score.
        """
        ...

    @abstractmethod
    def gap(self, n: int = 1) -> float:
        """Score an insertion or deletion (gap).

        :parm n: Size of gap.

        :returns: Gap score for one operation.
        """
        ...

    @abstractmethod
    def template_switch(self, n: int = 1) -> float:
        """Score a template switch.

        :returns: Template switch score.
        """
        ...

    @abstractmethod
    def off_variant(self) -> float:
        """Multiplier for off-variant features.

        For scored features that are incompatible with a variant type, for example reference gaps
        at an insertion site, multiply the gap score by this factor.

        :returns: Multiplier for off-variant features.
        """
        ...

    def score_op(self, op_code: int | str, op_len: int) -> float:
        """Score one alignment operation.

        :param op_code: Operation code as a operation code integer or a capitalized operation symbol.
        :param op_len: Operation length.

        :returns: Score for one alignment operation. Returns 0.0 if the operation is not scored (S, H, N, and P
            operations).
        """
        if op_code in {'=', op.EQ}:
            return self.match(op_len)

        elif op_code in {'X', op.X}:
            return self.mismatch(op_len)

        elif op_code in {'I', 'D', op.I, op.D}:
            return self.gap(op_len)

        elif op_code in {'S', 'H', op.S, op.H}:
            return 0.0

        elif op_code in {'M', op.M}:
            raise RuntimeError('Cannot score alignments with match ("M") in CIGAR string (requires "=" and "X")')

        elif op_code in {'N', 'P', op.N, op.P}:
            return 0.0

        else:
            raise RuntimeError(f'Unrecognized alignment operation: {op_code}')

    def score_op_arr(self, op_arr: np.ndarray) -> float:
        """Score all operations in an array and sum.

        A vectorized implementation of summing scores for affine models.

        :param op_arr: Array of alignment operations (op_code: first column, op_len: second column).

        :returns: Sum of alignment scores across all operations.
        """
        return np.sum(np.vectorize(self.score_op)(op_arr[:, 0], op_arr[:, 1]))

    def score_align_table(self, df: pl.DataFrame) -> pl.Series:
        """Score all alignment records in a table.

        :param df: Table of alignment records with "align_ops" column (list of structs with "op_code" and "op_len"
            fields).

        :returns: Sum of alignment scores across all operations for each record.
        """
        return (
            df
            .select(pl.col('align_ops'))
            .to_series()
            .map_elements(op.row_to_arr, return_dtype=pl.Object)
            .map_elements(lambda op_arr: np.float32(self.score_op_arr(op_arr)), return_dtype=pl.Float32)
        )

        # A pure polars implementation is slower:
        #
        # return (
        #     df
        #     .select(
        #         pl.col('align_ops')
        #         .map_elements(
        #             lambda ops_list: sum(
        #                 self.score_op(op['op_code'], op['op_len'])
        #                 for op in ops_list
        #             ),
        #             return_dtype=pl.Float32
        #         )
        #     )
        # )

    @abstractmethod
    def mismatch_model(self) -> 'ScoreModel':
        """Get a model for computing the score of mismatches.

        :returns A copy of this score model that does not penalize gaps.
        """
        ...

    @abstractmethod
    def model_param_string(self) -> str:
        """Get a parameter string representing this model.

        :returns: A parameter string for this score model.
        """
        ...

    @abstractmethod
    def __eq__(self, other: 'ScoreModel') -> bool:
        """Check equality with another :class:`ScoreModel`."""
        ...

    def __repr__(self) -> str:
        """Get a string representation of the model."""
        return 'ScoreModel(Interface)'


@immutable
class AffineScoreModel(ScoreModel):
    """Affine score model with default values modeled on minimap2 (2.26) default parameters.

    :param score_match: Match score [2].
    :param score_mismatch: Mismatch penalty [4].
    :param score_affine_gap: A list of tuples of two elements (gap open, gap extend) penalty.
    :param score_template_switch: Template switch penalty. If none, defaults to 2x the penalty of a 50 bp gap.
    """

    score_match: float
    score_mismatch: float
    score_affine_gap: tuple[tuple[float, float], ...]
    score_template_switch: float
    mul_off_variant: float

    def __init__(
            self,
            score_match: float = AFFINE_SCORE_MATCH,
            score_mismatch: float = AFFINE_SCORE_MISMATCH,
            score_affine_gap: tuple[tuple[float, float], ...] = AFFINE_SCORE_GAP,
            score_template_switch: float | tuple[float, int] = AFFINE_SCORE_TS,
            mul_off_variant: float = AFFINE_OFF_VARIANT_PENALTY,
    ):
        self.score_match = abs(float(score_match))
        self.score_mismatch = -abs(float(score_mismatch))
        self.score_affine_gap = tuple(
            (-abs(float(gap_open)), -abs(float(gap_extend)))
            for gap_open, gap_extend in score_affine_gap
        )

        if isinstance(score_template_switch, tuple):
            if len(score_template_switch) != 2:
                raise ValueError(f'Template switch score tuple must be 2 elements (multiplier, gap size): {score_template_switch}')

            self.score_template_switch = (
                abs(float(score_template_switch[0]))
                * self.gap(abs(int(score_template_switch[1])))
            )
        else:
            self.score_template_switch = -abs(float(score_template_switch))

        self.mul_off_variant = abs(float(mul_off_variant))

        if self.mul_off_variant <= 1.0:
            raise ValueError(f'Off-target variant multiplier must be > 1.0: {self.mul_off_variant}')

    def match(self, n: int = 1) -> float:
        """Score match.

        :param n: Number of matching bases.

        :returns: Match score.
        """
        return self.score_match * n

    def mismatch(self, n: int = 1) -> float:
        """Score mismatch.

        :param n: Number of mismatched bases.

        :returns: Mismatch score.
        """
        return self.score_mismatch * n

    def gap(self, n: int = 1) -> float:
        """Score a gap (insertion or deletion).

        Compute all affine alignment gap scores and return the lowest penalty.

        :param n: Size of gap.

        :returns: Gap score.
        """  # noqa: D402
        if n == 0.0:
            return 0.0

        return max([
            gap_open + (gap_extend * n)
            for gap_open, gap_extend in self.score_affine_gap
        ])

    def template_switch(self, n: int = 1) -> float:
        """Score a template switch.

        :param n: Number of template switches.

        :returns: Template switch score.
        """
        return self.score_template_switch * n

    def off_variant(self) -> float:
        """Multiplier for off-variant features.

        For scored features that are incompatible with a variant type, for example reference gaps
        at an insertion site, multiply the gap score by this factor.

        :returns: Multiplier for off-variant features.
        """
        return self.mul_off_variant

    def mismatch_model(self) -> 'ScoreModel':
        """Create a version of this model that scores only mismatches (ignores gaps).

        The mismatch model retains the template-switch penalty based on the original model
        even if it was derived from gap scores. A new model is returned and this model is not altered.

        :returns: A copy of this score model that does not penalize gaps.
        """
        return AffineScoreModel(
            score_match=self.score_match,
            score_mismatch=self.score_mismatch,
            score_affine_gap=((0.0, 0.0),),
            score_template_switch=self.score_template_switch,
            mul_off_variant=self.mul_off_variant,
        )

    def score_operations(
            self,
            op_arr: np.ndarray
    ) -> float:
        """Get a vectorized implementation of summing scores for affine models.

        :param op_arr: Array of alignment operations (op_code: first column, op_len: second column).

        :returns: Sum of scores across all alignment operations.
        """
        if np.any(op_arr[:, 0] == op.M):
            raise RuntimeError('Cannot score alignments with match ("M") in CIGAR string (requires "=" and "X")')

        # Score gaps
        gap_arr = op_arr[(op_arr[:, 0] == op.D) | (op_arr[:, 0] == op.I), 1]

        gap_score = np.full((gap_arr.shape[0], 2), -np.inf)

        for gap_open, gap_extend in self.score_affine_gap:
            gap_score[:, 1] = gap_open + gap_arr * gap_extend
            gap_score[:, 0] = np.max(gap_score, axis=1)

        return (
            np.sum(op_arr[:, 1] * (op_arr[:, 0] == op.EQ) * self.score_match)
            + np.sum(op_arr[:, 1] * (op_arr[:, 0] == op.X) * self.score_mismatch)
            + np.nan_to_num(gap_score[:, 0], neginf=0.0).sum()  # If no gap penalties (i.e. mismatch model), then gap_score is -inf (set to 0.0)
        )

    def model_param_string(self) -> str:
        """Get a parameter string representing this model.

        :returns: A parameter string for this score model.
        """
        gap_str = ";".join([f"{gap_open}:{gap_extend}" for gap_open, gap_extend in self.score_affine_gap])

        return (
            f'affine::'
            f'match={self.score_match},'
            f'mismatch={self.score_mismatch},'
            f'gap={gap_str},'
            f'ts={self.score_template_switch},'
            f'off_var={self.mul_off_variant}'
        )

    def __eq__(self, other: 'ScoreModel') -> bool:
        """Check equality with another :class:`ScoreModel`."""
        if other is None or not isinstance(other, self.__class__):
            return False

        return (
            self.score_match == other.score_match and
            self.score_mismatch == other.score_mismatch and
            self.score_affine_gap == other.score_affine_gap and
            self.score_template_switch == other.score_template_switch
        )

    def __repr__(self) -> str:
        """Get a string representation of the model."""
        gap_str = ';'.join([f'{abs(gap_open)}:{abs(gap_extend)}' for gap_open, gap_extend in self.score_affine_gap])
        return (f'AffineScoreModel(match={self.score_match},mismatch={-self.score_mismatch},'
                f'gap={gap_str},ts={-self.score_template_switch})')


def get_score_model(
        param_string: Optional[str | ScoreModel] = None
) -> ScoreModel:
    """Get score model from a string of alignment parameters.

    :param param_string: Parameter string. May be None or an empty string (default model is used). If the string is
        an instance of `ScoreModel`, then the `ScoreModel` object is returned. Otherwise, the string is parsed and
        a score model object is returned.

    :returns: A :class:`ScoreModel` object.

    :raises ValueError: If the string cannot be parsed or the model type is not recognized.
    """
    if isinstance(param_string, ScoreModel):
        return param_string

    if param_string is not None:
        param_string = param_string.strip()

    if param_string is None or len(param_string) == 0:
        param_string = DEFAULT_ALIGN_SCORE_MODEL

    if '::' in param_string:
        model_type, model_params = re.split(r'::', param_string, maxsplit=1)

    else:
        model_type, model_params = 'affine', param_string

    if model_type == 'affine':
        return get_affine_by_params(model_params)

    raise ValueError(f'Unrecognized score model type: {model_type}')


def get_affine_by_params(param_string: str) -> AffineScoreModel:
    """Parse a string to get alignment parameters from it.

    :param param_string: Parameter string.

    :returns: A configured AffineScoreModel.

    :raises ValueError: If the string cannot be parsed.
    """
    # Set defaults
    params: dict[str, Any] = {
        'score_match': AFFINE_SCORE_MATCH,
        'score_mismatch': AFFINE_SCORE_MISMATCH,
        'score_affine_gap': AFFINE_SCORE_GAP,
        'score_template_switch': AFFINE_SCORE_TS,
        'mul_off_variant': AFFINE_OFF_VARIANT_PENALTY,
    }

    key_to_param = {
        'match': 'score_match',
        'mismatch': 'score_mismatch',
        'gap': 'score_affine_gap',
        'ts': 'score_template_switch',
        'off_var': 'mul_off_variant',
    }

    keys = list(key_to_param.keys())

    # Sanitize parameter string
    if param_string is not None:
        param_string = param_string.strip()

        if len(param_string) == 0:
            param_string = None

    if param_string is None:
        param_string = DEFAULT_ALIGN_SCORE_MODEL

    # Parse param string
    param_pos = 0

    for tok in param_string.split(','):
        tok = tok.strip()

        if len(tok) == 0:
            param_pos += 1
            continue  # Accept default for missing

        if '=' in tok:
            param_pos = None  # Do not allow positional parameters after named ones

            key, val = tok.split('=', 1)

            key = key.strip()
            val = val.strip()

        else:
            if param_pos is None:
                raise ValueError(
                    f'Named parameters (with "=") must be specified after positional parameters (no "="): '
                    f'{param_string}'
                )

            if param_pos >= len(keys):
                raise ValueError(f'Too many positional parameters in: {param_string}')

            key = keys[param_pos]
            val = tok

            param_pos += 1

        if not val:
            continue

        try:
            param_name = key_to_param[key]
        except KeyError:
            raise ValueError(f'Unrecognized parameter key: {key}')

        if key == 'gap':
            gap_list = []

            for gap_pair in val.split(';'):
                gap_tok = gap_pair.split(':')

                if len(gap_tok) != 2:
                    raise ValueError(f'Invalid gap format: Expected "open:extend" for each element '
                                     f'(multiple pairs separated by ";"): "{gap_pair}" in {val}')

                try:
                    gap_open = abs(float(gap_tok[0].strip()))
                    gap_extend = abs(float(gap_tok[1].strip()))

                except ValueError as e:
                    raise ValueError(f'Invalid gap format: Expected numeric values for "open:extend" '
                                     f'for each gap cost element: {val}') from e

                gap_list.append((gap_open, gap_extend))

            val = tuple(gap_list)

        elif key == 'ts' and val[0] == '(' and val[-1] == ')':
            ts_tuple = val[1:-1].split(':')

            if len(ts_tuple) != 2:
                raise ValueError(f'Key "ts" requires a set value or a tuple[float, int]: Found {val}')

            val = (
                float(ts_tuple[0]),
                int(ts_tuple[1]),
            )

        else:
            try:
                val = abs(float(val))
            except ValueError as e:
                raise ValueError(f'Invalid numeric value for parameter "{key}": {val}') from e


        params[param_name] = val


    # Return alignment object
    return AffineScoreModel(
        **params
    )
