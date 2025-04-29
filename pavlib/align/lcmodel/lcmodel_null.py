"""
Null low-confidence alignment model. Accepts all alignments.
"""

import numpy as np
import pandas as pd

from . import score

from .lcmodel import LCAlignModel

class LCAlignModelNull(LCAlignModel):
    """
    Null model predicts no low-confidence alignments.
    """

    def __init__(self, lc_model_def=None):
        """
        Create a null model.

        :param lc_model_def: Ignored.
        """

        if lc_model_def is None:
            lc_model_def = {
                'name': 'null',
                'type': 'null',
                'description': 'Null model, predicts no low-confidence alignments'
            }

        super().__init__(lc_model_def)

    def __call__(self,
                 df: pd.DataFrame,
                 existing_score_model: score.ScoreModel=None,
                 op_arr_list: list[np.ndarray]=None,
                 qry_fai: pd.Series=None
        ) -> np.ndarray:
        """
        Predict low-confidence alignments.

        :param df: PAV Alignment table.
        :param existing_score_model: Existing score model used to compute features already in the alignment table (df).
            If this alignment score model matches the alignment score model used to train this LC model, then features
            are re-used instead of re-computed.
        :param op_arr_list: A list of operation matrices (op_code: first column, op_len: second column) for each query
            record in `df`. If `None` and alignment operations are needed, they are generated from the CIGAR column of
            `df`.
        :param qry_fai: Query FASTA index. Needed if features need to be computed using the full query sequence size
            (i.e. QRY_PROP).

        :return: Boolean array of predicted low-confidence alignments.
        """

        return np.repeat(False, df.shape[0])
