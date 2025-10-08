"""Logistic low-confindence alignment model."""

__all__ = [
    'LCAlignModelLogistic',
]

from dataclasses import dataclass
from typing import Callable, Optional

import frozendict
import numpy as np
import polars as pl
import scipy.special

from ..score import ScoreModel

from ._lcmodel import LCAlignModel


@dataclass(frozen=True, repr=False)
class LCAlignModelLogistic(LCAlignModel):
    """Use a pre-trained logistic regression model to predict low-confidence alignments."""

    def __post_init__(self):
        """Check model invariants."""
        super().__post_init__()

        # Copy model definition
        lc_model_def = dict(self.lc_model_def)

        # Check and set
        lc_model_def['threshold'] = float(lc_model_def.get('threshold', 0.5))

        if not 0.0 <= lc_model_def['threshold'] <= 1.0:
            raise ValueError(
                f'LC align model {self.name} threshold attribute must be in range [0.0, 1.0]: '
                f'{lc_model_def["threshold"]}'
            )

        lc_model_def['weight_filename'] = str(lc_model_def.get('weight_filename', 'weights.npz'))

        with self.resource_reader(lc_model_def['weight_filename'], text_mode=False) as in_file:
            loader = np.load(in_file)

            lc_model_def['w'] = loader['w']
            lc_model_def['b'] = loader['b']

        # Freeze model definition
        object.__setattr__(self, 'lc_model_def', frozendict.frozendict(lc_model_def))

        # Check for unknown attributes
        self.check_unknown_attributes()

    @property
    def activation(self) -> Callable[[np.ndarray], np.ndarray]:
        """Activation function."""
        return scipy.special.expit

    @property
    def threshold(self) -> float:
        """Low-confidence threshold."""
        return self.lc_model_def['threshold']

    @property
    def weight_filename(self) -> str:
        """Model weight file location."""
        return self.lc_model_def['weight_filename']

    @property
    def w(self) -> np.ndarray:
        """Feature weight array."""
        return self.lc_model_def['w']

    @property
    def b(self) -> np.ndarray:
        """Model bias term."""
        return self.lc_model_def['b']

    def __call__(
            self,
            df: pl.DataFrame,
            existing_score_model: Optional[ScoreModel | str] = None,
            df_qry_fai: Optional[pl.Series | str] = None
    ) -> np.ndarray:
        """Predict low-confidence alignments.

        :param df: PAV Alignment table.
        :param existing_score_model: Existing score model used to compute features already in the alignment table (df).
            If this alignment score model matches the alignment score model used to train this LC model, then
            features are re-used instead of re-computed.
        :param df_qry_fai: Query FASTA index. Needed if features need to be computed using the full query sequence size.

        :returns: Boolean array of predicted low-confidence alignments.
        """
        return self.activation(
            (
                self.get_feature_table(
                    df=df,
                    existing_score_model=existing_score_model,
                    df_qry_fai=df_qry_fai
                )
                .cast(pl.Float32)
                .to_numpy()
            ) @ self.w + self.b
        ).reshape(-1) >= self.threshold
