"""
Alignment routines.
"""

import abc
import numpy as np
import re

from . import op

AFFINE_SCORE_MATCH = 2.0
AFFINE_SCORE_MISMATCH = 4.0
AFFINE_SCORE_GAP = ((4.0, 2.0), (24.0, 1.0))
AFFINE_SCORE_TS = None

DEFAULT_ALIGN_SCORE_MODEL = f'affine::match={AFFINE_SCORE_MATCH},mismatch={AFFINE_SCORE_MISMATCH},gap={";".join([f"{gap_open}:{gap_extend}" for gap_open, gap_extend in AFFINE_SCORE_GAP])}'

class ScoreModel(object, metaclass=abc.ABCMeta):
    """
    Score model interface.
    """

    @abc.abstractmethod
    def match(self, n=1):
        """
        Score match.

        :param n: Number of matching bases.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def mismatch(self, n=1):
        """
        Score mismatch.

        :param n: Number of mismatched bases.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def gap(self, n=1):
        """
        Score gap (insertion or deletion). Compute all affine aligment gap scores and return the lowest.

        :param n: Size of gap.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def template_switch(self):
        """
        Score a template switch.

        :return: Template switch score.
        """

        raise NotImplementedError

    def score(self, op_code, op_len):
        """
        Score a CIGAR operation and return 0 if the operation is not scored (S, H, N, and P CIGAR operations). The
        CIGAR operations may be a capital letter or symbol (e.g. "=", "X", "I", etc.) on the numeric CIGAR operation
        code (defined in `pavlib.align.util`).

        :param op_len: CIGAR operation length.
        :param op_code: CIGAR operation code, must be capitalized.
        """

        if op_code in {'=', op.EQ}:
            return self.match(op_len)

        elif op_code in {'X', op.X}:
            return self.mismatch(op_len)

        elif op_code in {'I', 'D', op.I, op.D}:
            return self.gap(op_len)

        elif op_code in {'S', 'H', op.S, op.H}:
            return 0

        elif op_code in {'M', op.M}:
            raise RuntimeError('Cannot score alignments with match ("M") in CIGAR string (requires "=" and "X")')

        elif op_code in {'N', 'P', op.N, op.P}:
            return 0

        else:
            raise RuntimeError(f'Unrecognized CIGAR op code: {op_code}')

    def score_operations(self,
                         op_arr: np.ndarray,
                         ) -> float:
        """
        A vectorized implementation of summing scores for affine models.

        :param op_arr: Array of alignment operations (op_code: first column, op_len: second column).

        :return: Sum of scores for each CIGAR operation.
        """

        return np.sum(np.vectorize(self.score)(op_arr[:, 0], op_arr[:, 1]))

    @abc.abstractmethod
    def mismatch_model(self):
        """
        Return a copy of this score model that does not penalize gaps. Used for computing the score of mismatches.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def model_param_string(self):
        """
        Return a parameter string for this score model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        return 'ScoreModel(Interface)'


class AffineScoreModel(ScoreModel):
    score_match: float
    score_mismatch: float
    score_affine_gap: tuple
    score_template_switch: float
    """
    Affine score model with default values modeled on minimap2 (2.26) default parameters.

    :param match: Match score [2].
    :param mismatch: Mismatch penalty [4].
    :param affine_gap: A list of tuples of two elements (gap open, gap extend) penalty.
    :param template_switch: Template switch penalty. If none, defaults to 2x the penalty of a 50 bp gap.
    """

    def __init__(self,
             match: float=AFFINE_SCORE_MATCH,
             mismatch: float=AFFINE_SCORE_MISMATCH,
             affine_gap: float=AFFINE_SCORE_GAP,
             template_switch: float=AFFINE_SCORE_TS
         ):
        self.score_match = np.abs(match)
        self.score_mismatch = -np.abs(mismatch)
        self.score_affine_gap = tuple((
            (-np.abs(gap_open), -np.abs(gap_extend))
                for gap_open, gap_extend in affine_gap
        ))

        if template_switch is None:
            self.score_template_switch = 2 * self.gap(50)
        else:
            try:
                self.score_template_switch = - np.abs(float(template_switch))
            except ValueError:
                raise ValueError(f'template_switch parameter is not numeric: {template_switch}')

    def match(self, n: int=1) -> float:
        """
        Score match.

        :param n: Number of matching bases.
        """

        return self.score_match * n

    def mismatch(self, n: int=1) -> float:
        """
        Score mismatch.

        :param n: Number of mismatched bases.
        """

        return self.score_mismatch * n

    def gap(self, n: int=1) -> float:
        """
        Score gap (insertion or deletion). Compute all affine alignment gap scores and return the lowest penalty.

        :param n: Size of gap.
        """

        if n == 0.0:
            return 0.0

        return np.max(
            [
                gap_open + (gap_extend * n)
                    for gap_open, gap_extend in self.score_affine_gap
            ]
        )

    def template_switch(self) -> float:
        """
        Score a template switch.

        :return: Template switch score.
        """
        return self.score_template_switch

    def mismatch_model(self) -> ScoreModel:
        """
        Create a version of this model that scores only mismatches (ignores gaps). Thd mismatch model retains the
        template-switch penalty based on the original model even if it was derived from gap scores. A new model
        is returned and this model is not altered.
        """
        return AffineScoreModel(
            match=self.score_match,
            mismatch=self.score_mismatch,
            affine_gap=[(0.0, 0.0)],
            template_switch=self.score_template_switch
        )

    def score_operations(self,
                         op_arr: np.ndarray
                         ) -> float:
        """
        A vectorized implementation of summing scores for affine models.

        :param op_arr: Array of alignment operations (op_code: first column, op_len: second column).

        :return: Sum of scores for each CIGAR operation.
        """

        if np.any(op_arr[:, 0] == op.M):
            raise RuntimeError('Cannot score alignments with match ("M") in CIGAR string (requires "=" and "X")')

        # Score gaps
        gap_arr = op_arr[(op_arr[:, 0] == op.D) | (op_arr[:, 0] == op.I), 1]

        gap_score = np.full((gap_arr.shape[0], 2), -np.inf)

        for gap_open, gap_extend in self.score_affine_gap:
            gap_score[:, 1] = gap_open + gap_arr * gap_extend
            gap_score[:, 0] = np.max(gap_score, axis=1)

        return \
                np.sum(op_arr[:, 1] * (op_arr[:, 0] == op.EQ) * self.score_match) + \
                np.sum(op_arr[:, 1] * (op_arr[:, 0] == op.X) * self.score_mismatch) + \
                np.nan_to_num(gap_score[:, 0], neginf=0.0).sum()  # If no gap penalties (i.e. mismatch model), then gap_score is -inf (set to 0.0)

    def model_param_string(self):
        return f'affine::match={self.score_match},mismatch={self.score_mismatch},gap={";".join([f"{gap_open}:{gap_extend}" for gap_open, gap_extend in self.score_affine_gap])}'

    def __eq__(self, other):

        if other is None or not isinstance(other, self.__class__):
            return False

        return \
                self.score_match == other.score_match and \
                self.score_mismatch == other.score_mismatch and \
                self.score_affine_gap == other.score_affine_gap and \
                self.score_template_switch == other.score_template_switch

    def __repr__(self):
        gap_str = ';'.join([f'{abs(gap_open)}:{abs(gap_extend)}' for gap_open, gap_extend in self.score_affine_gap])

        return f'AffineScoreModel(match={self.score_match},mismatch={-self.score_mismatch},gap={gap_str},ts={-self.score_template_switch})'


def get_score_model(
        param_string: str=None
) -> ScoreModel:
    """
    Get score model from a string of alignment parameters.

    :param param_string: Parameter string. May be None or an empty string (default model is used). If the string is
        an instance of `ScoreModel`, then the `ScoreModel` object is returned. Otherwise, the string is parsed and
        a score model object is returned.

    :return: A `ScoreModel` object.
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

    raise RuntimeError(f'Unrecognized score model type: {model_type}')

def get_affine_by_params(
        param_string: str
) -> AffineScoreModel:
    """
    Parse a string to get alignment parameters from it.

    :param param_string: Parameter string.

    :return: A configured AffineScoreModel.
    """

    # Set defaults
    params = [
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

            key, val = tok.split('=')

            key = key.strip()
            val = val.strip()

        else:
            if param_pos is None:
                raise RuntimeError(f'Named parameters (with "=") must be specified after positional parameters (no "="): {param_string}')

            key = keys[param_pos]
            val = tok

            param_pos += 1

        if key in {'match', 'mismatch', 'ts'}:
            try:
                val = abs(float(val))
            except ValueError:
                raise RuntimeError(f'Unrecognized alignment parameter: {key} (allowed: match, mismatch, gap, ts)')

        if key == 'match':
            params[0] = val

        elif key == 'mismatch':
            params[1] = val

        elif key == 'gap':

            gap_list = list()

            for gap_pair in val.split(';'):
                gap_tok = gap_pair.split(':')

                if len(gap_tok) != 2:
                    raise RuntimeError(f'Unrecognized gap format: Expected "open:extend" for each element (multiple pairs separated by ";"): "{gap_pair}" in {val}')

                try:
                    gap_open = abs(float(gap_tok[0].strip()))
                    gap_extend = abs(float(gap_tok[1].strip()))

                except ValueError:
                    raise RuntimeError(f'Unrecognized gap format: Expected integer values for "open:extend" for each gap cost element: {val}')

                gap_list.append((gap_open, gap_extend))

            params[2] = tuple(gap_list)

        elif key == 'ts':
            params[3] = val

        else:
            raise RuntimeError(f'Unrecognized alignment parameter: {key} (allowed: match, mismatch, gap, ts)')

    # Return alignment object
    return AffineScoreModel(*params)
