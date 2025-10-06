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

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
from typing import Any, Optional

import numpy as np
import polars as pl

from . import op

AFFINE_SCORE_MATCH: float = 2.0
"""Match score (minimap2 default)"""

AFFINE_SCORE_MISMATCH: float = 4.0
"""Mismatch score (minimap2 default)"""

AFFINE_SCORE_GAP: tuple[tuple[float, float], ...] = ((4.0, 2.0), (24.0, 1.0))
"""Multi-affine gap scores as a sequence of gap-open and gap-extend penalties (minimap2 defaults)."""

AFFINE_SCORE_TS = None
"""Template switch score. If None, defaults to 2x the penalty of a 50 bp gap."""

DEFAULT_ALIGN_SCORE_MODEL = (
    f'affine::'
    f'match={AFFINE_SCORE_MATCH},'
    f'mismatch={AFFINE_SCORE_MISMATCH},'
    f'gap={";".join([f"{gap_open}:{gap_extend}" for gap_open, gap_extend in AFFINE_SCORE_GAP])}'
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
        pass

    @abstractmethod
    def mismatch(self, n: int = 1) -> float:
        """Score mismatched bases.

        :param n: Number of mismatched bases.

        :returns: Mismatch score.
        """
        pass

    @abstractmethod
    def gap(self, n: int = 1) -> float:
        """Score an insertion or deletion (gap).

        :parm n: Size of gap.

        :returns: Gap score for one operation.
        """
        pass

    @abstractmethod
    def template_switch(self) -> float:
        """Score a template switch.

        :returns: Template switch score.
        """
        pass

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
            .map_elements(self.score_op_arr, return_dtype=pl.Float32)
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
        pass

    @abstractmethod
    def model_param_string(self) -> str:
        """Get a parameter string representing this model.

        :returns: A parameter string for this score model.
        """
        pass

    @abstractmethod
    def __eq__(self, other: 'ScoreModel') -> bool:
        """Check equality with another :class:`ScoreModel`."""
        pass

    def __repr__(self) -> str:
        """Get a string representation of the model."""
        return 'ScoreModel(Interface)'


@dataclass(frozen=True)
class AffineScoreModel(ScoreModel):
    """Affine score model with default values modeled on minimap2 (2.26) default parameters.

    :param score_match: Match score [2].
    :param score_mismatch: Mismatch penalty [4].
    :param score_affine_gap: A list of tuples of two elements (gap open, gap extend) penalty.
    :param score_template_switch: Template switch penalty. If none, defaults to 2x the penalty of a 50 bp gap.
    """

    score_match: float = field(default=AFFINE_SCORE_MATCH)
    score_mismatch: float = field(default=AFFINE_SCORE_MISMATCH)
    score_affine_gap: tuple[tuple[float, float], ...] = field(default=AFFINE_SCORE_GAP)
    score_template_switch: float | None = field(default=AFFINE_SCORE_TS)

    def __post_init__(self):
        """Ensure penalties are set correctly."""
        object.__setattr__(self, 'score_match', float(abs(self.score_match)))
        object.__setattr__(self, 'score_mismatch', float(-abs(self.score_mismatch)))
        object.__setattr__(self, 'score_affine_gap', tuple((
            (float(-abs(gap_open)), float(-abs(gap_extend)))
            for gap_open, gap_extend in self.score_affine_gap
        )))

        if self.score_template_switch is None:
            object.__setattr__(self, 'score_template_switch', 2 * self.gap(50))
        else:
            object.__setattr__(self, 'score_template_switch', float(-abs(self.score_template_switch)))

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

    def template_switch(self) -> float:
        """Score a template switch.

        :returns: Template switch score.
        """
        return self.score_template_switch

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
            score_template_switch=0.0
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
            np.sum(op_arr[:, 1] * (op_arr[:, 0] == op.EQ) * self.score_match) +
            np.sum(op_arr[:, 1] * (op_arr[:, 0] == op.X) * self.score_mismatch) +
            # If no gap penalties (i.e. mismatch model), then gap_score is -inf (set to 0.0)
            np.nan_to_num(gap_score[:, 0], neginf=0.0).sum()
        )

    def model_param_string(self) -> str:
        """Get a parameter string representing this model.

        :returns: A parameter string for this score model.
        """
        gap_str = ";".join([f"{gap_open}:{gap_extend}" for gap_open, gap_extend in self.score_affine_gap])

        return f'affine::match={self.score_match},mismatch={self.score_mismatch},gap={gap_str}'

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
    params: list[Any] = [
        AFFINE_SCORE_MATCH,
        AFFINE_SCORE_MISMATCH,
        AFFINE_SCORE_GAP,
        AFFINE_SCORE_TS
    ]

    keys = ['match', 'mismatch', 'gap', 'ts']

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

        if key in {'match', 'mismatch', 'ts'}:
            try:
                val = abs(float(val))
            except ValueError as e:
                raise ValueError(f'Invalid numeric value for parameter "{key}": {val}') from e

        if key == 'match':
            params[0] = val

        elif key == 'mismatch':
            params[1] = val

        elif key == 'gap':
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

            params[2] = tuple(gap_list)

        elif key == 'ts':
            params[3] = val

        else:
            raise ValueError(f'Unrecognized alignment parameter: {key} (allowed: match, mismatch, gap, ts)')

    # Return alignment object
    return AffineScoreModel(
        score_match=params[0],
        score_mismatch=params[1],
        score_affine_gap=params[2],
        score_template_switch=params[3]
    )
