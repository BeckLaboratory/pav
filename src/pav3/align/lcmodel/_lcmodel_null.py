"""Null low-confidence alignment model.

Accepts all alignments.
"""

__all__ = [
    'LCAlignModelNull'
]

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from ..score import ScoreModel

from ._lcmodel import LCAlignModel


@dataclass(frozen=True)
class LCAlignModelNull(LCAlignModel):
    """Null model predicts no low-confidence alignments."""

    def __post_init__(self):
        """Post-initialization.

        :raises ValueError: Unknown attributes are found.
        """
        object.__setattr__(self, 'lc_model_def', {
            'name': 'null',
            'type': 'null',
            'description': 'Null model, predicts no low-confidence alignments'
        })

        super().__post_init__()
        self.check_unknown_attributes()

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
        return np.repeat(False, df.shape[0])
