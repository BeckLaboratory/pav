"""
Logistic model implementation.
"""

import numpy as np
import os
import scipy.special

from .lcmodel import LCAlignModel

class LCAlignModelLogistic(LCAlignModel):
    """
    Use a pre-trained logistic regression model to predict low-confidence alignments.
    """
    def __init__(self, lc_model_def):
        """
        Create model.

        :param lc_model_def: LC align model definition dictionary.
        """
        super().__init__(lc_model_def)

        self.activation = scipy.special.expit

        # Set threshold
        self.threshold = self.lc_model_def.get('threshold', 0.5)
        self.known_attr.add('threshold')

        try:
            self.threshold = float(self.threshold)
        except ValueError:
            raise RuntimeError(f'LC align model {self.name} threshold attribute must be a numeric value: {self.threshold} (type="{type(self.threshold)}")')

        if self.threshold < 0.0 or self.threshold > 1.0:
            raise RuntimeError(f'LC align model {self.name} threshold attribute must be in range [0.0, 1.0]: {self.threshold:.2f}')

        self.known_attr.add('threshold')

        # Load weights
        weight_filename = os.path.join(self.model_path, 'weights.npz')

        if not os.path.exists(weight_filename):
            raise RuntimeError(f'LC align model {self.name} weight file not found: {weight_filename}')

        loader = np.load(weight_filename)

        missing_keys = [key for key in ['w', 'b'] if key not in loader.keys()]

        if missing_keys:
            raise RuntimeError(f'LC align model {self.name} weight file "{weight_filename}" is missing required keys: {", ".join(missing_keys)}')

        self.w = loader['w']
        self.b = loader['b']

        # Check for unknown attributes
        self.check_known_attributes()


    def __call__(self, df, existing_score_model=None, qry_fai=None):
        """
        Predict low-confidence alignments.

        :param df: PAV Alignment table.
        :param existing_score_model: Existing score model used to compute features already in the alignment table (df).
            If this alignment score model matches the alignment score model used to train this LC model, then features
            are re-used instead of re-computed.
        :param qry_fai: Query FASTA index. Needed if features need to be computed using the full query sequence size
            (i.e. QRY_PROP).

        :return: Boolean array of predicted low-confidence alignments.
        """

        return self.activation(
            self.get_feature_table(df, existing_score_model, qry_fai).values.astype(float) @ self.w + self.b
        ).reshape(-1) >= self.threshold
