"""
Null low-confidence alignment model. Accepts all alignments.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from . import score

from .lcmodel import LCAlignModel

@dataclass(frozen=True)
class LCAlignModelNull(LCAlignModel):
    """
    Null model predicts no low-confidence alignments.
    """

    def __post_init__(self):
        """
        Post-initialization.

        Raises:
            ValueError: Unknown attributes are found.
        """

        if self.lc_model_def is None:
            object.__setattr__(self, 'lc_model_def', {
                'name': 'null',
                'type': 'null',
                'description': 'Null model, predicts no low-confidence alignments'
            })

        super().__post_init__()
        self.check_unknown_attributes()

    def __call__(self,
                 df: pl.DataFrame,
                 existing_score_model: Optional[score.ScoreModel | str] = None,
                 df_qry_fai: Optional[pl.Series | str] = None
        ) -> np.ndarray:
        """
        Predict low-confidence alignments.

        Args:
            df: PAV Alignment table.
            existing_score_model: Existing score model used to compute features already in the alignment table (df).
                If this alignment score model matches the alignment score model used to train this LC model, then
                features are re-used instead of re-computed.
            df_qry_fai: Query FASTA index. Needed if features need to be computed using the full query sequence size.

        Returns:
            Boolean array of predicted low-confidence alignments.
        """

        return np.repeat(False, df.shape[0])
