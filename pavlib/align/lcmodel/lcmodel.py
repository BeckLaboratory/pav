"""
A model for flagging low-confidence (LC) alignment records.
"""

import abc
import numpy as np
import pandas as pd

import svpoplib

from .. import score
from .. import features


"""
Example model JSON:

{
    "name": "default",
    "description": "Low-confidence alignment model trained on HGSVC assemblies (Logsdon 2025)",
    "type": "logistic",
    "type_version": 0,
    "features": [
        "SCORE_PROP", "SCORE_MM_PROP", "ANCHOR_PROP", "QRY_PROP"
    ],
    "threshold": 0.5,
    "score_model": "affine::match=2.0,mismatch=4.0,gap=4.0:2.0;24.0:1.0",
    "score_prop_conf": 0.85
}
"""

class LCAlignModel(object, metaclass=abc.ABCMeta):
    """
    Base class for a model used to determine low-confidence (LC) alignment records from a PAV alignment BED file. This
    class takes a model definition and provides methods for generating features and making LC predictions.
    """

    def __init__(self, lc_model_def):
        """
        Create a model for predicting low-confidence alignment records.

        :param lc_model_def: LC model definition dictionary from a model definition JSON file.
        """

        # Track known attributes as they are processed. This allows child classes to flag unknown parameters that are
        # not handled by parent classes. Nested attributes use slashes: e.g. "parameters/score_model" in this set refers
        # indicates "lc_model_def['parameters']['score_model']" was processed.
        self.known_attr = set()

        # Model definition
        if lc_model_def is None or not isinstance(lc_model_def, dict):
            raise RuntimeError(f'LC align model definition is not a dictionary: {type(lc_model_def)}')

        self.lc_model_def = lc_model_def

        if 'type' not in lc_model_def:
            raise RuntimeError(f'LC align model definition is missing the "type" attribute')

        self.model_type = lc_model_def['type']
        self.known_attr.add('type')

        self.type_version = lc_model_def.get('type_version', 0)
        self.known_attr.add('type_version')

        self.name = lc_model_def.get('name', '<MODEL_NAME_NOT_SPECIFIED>')
        self.known_attr.add('name')

        self.description = lc_model_def.get('description', None)
        self.known_attr.add('description')

        self.model_path = lc_model_def.get('model_path', None)
        self.known_attr.add('model_path')

        # Feature array
        if len(lc_model_def.get('features', [])) == 0 and self.model_type != 'null':
            raise RuntimeError(f'LCAlignModel is missing features in the "features" attribute: {self.name}')

        self.feature_list = lc_model_def.get('features', [])

        if not isinstance(self.feature_list, list):
            raise RuntimeError(f'LCAlignModel features must be a list ("features" attribute): {self.name}')

        self.feature_list = [str(feature).strip() for feature in self.feature_list]

        if any([feature == '' for feature in self.feature_list]):
            raise RuntimeError(f'LCAlignModel features cannot be empty strings ("features" attribute): {self.name}')

        self.feature_count = len(self.feature_list)

        if len(set(self.feature_list)) != self.feature_count:
            raise RuntimeError(f'LCAlignModel features must be unique ("features" attribute): {self.name}')

        self.known_attr.add('features')

        # Alignment (CIGAR) score model used by this LCAlignModel
        self.score_model = self.lc_model_def.get('score_model', None)

        if isinstance(self.score_model, str):
            try:
                self.score_model = score.get_score_model(self.score_model.strip())
            except Exception as e:
                raise RuntimeError(f'Bad score model definition for LCAlignModel {self.name}: {e}')

        elif self.score_model is None:
            pass

        elif not isinstance(self.score_model, score.ScoreModel):
            raise RuntimeError(f'Bad score model definition for LCAlignModel {self.name}: expected a score model or a string specification: Unknown type "{type(self.score_model)}"')

        self.known_attr.add('score_model')

        # Known parameter attributes
        self.score_prop_conf = self.lc_model_def.get('score_prop_conf', None)

        if self.score_prop_conf is not None:
            try:
                self.score_prop_conf = float(self.score_prop_conf)
            except ValueError as e:
                raise RuntimeError(f'Parameter "score_prop_conf" in LC align model {self.name} is not a floating-point number: {self.score_prop_conf}')

            if self.score_prop_conf < 0.0 or self.score_prop_conf > 1.0:
                raise RuntimeError(f'Parameter "score_prop_conf" in LC align model {self.name} is not between 0.0 and 1.0: {self.score_prop_conf}')

        self.known_attr.add('score_prop_conf')  # Alignment score proportion for confident alignments based only on the score (not a trained score model).

        # Parameter for allowing unknown attributes
        try:
            self.allow_unknown_attributes = svpoplib.util.as_bool(self.lc_model_def.get('allow_unknown_attributes', False))
        except Exception as e:
            raise RuntimeError(f'Bad value for "allow_unknown_attributes" parameter for LCAlignModel {self.name}: {e}')

        self.known_attr.add('allow_unknown_attributes')

    def get_feature_table(self,
                          df: pd.DataFrame | pd.Series,
                          existing_score_model: str | score.ScoreModel | bool=None,
                          op_arr_list: np.ndarray[int, int]=None,
                          qry_fai: pd.Series=None
                          ):
        """
        Get a table of alignment features for a model.

        Scores for alignment records in `df` were created using tunable score model parameters, which may or may not
        match the score model that the LC model was trained on. Since recomputing scores requires nontrivial time, these
        scores are recomputed only if necessary.

        Existing features are recomputed only if necessary. For features requriing a score model, if both `score_model`
        and `existing_score_model` are specified, then alignment score features are only recomputed if they differ. If
        only `score_model` is specified (i.e. `existing_score_model` is None), then features requiring a score model
        are recomputed. If `score_model` is None and a feature requires a score model, an exception is raised.

        If features require the full length of query sequences (i.e. proportion of an assembly sequence aligned in each
        alignment record), then `qry_fai` is required to look up the full length of the query sequence. If these
        features are already present, then they are not recomputed and `qry_fai` is ignored.

        This method copies `df` and does not modify it.

        :param df: DataFrame or Series of alignment records. If a Series is given, then it is converted to a single-row
            DataFrame.
        :param existing_score_model: The score model used to compute features already in `df`. If this is not None and
            `score_model` is not compatible with it, then scores are recomputed. If None, assume existing scores do not
            need to be recomputed. As a special case, a boolean value of `True` forces re-computation, and `False` skips it.
            Value may be a string specification for a score model.
        :param op_arr_list: A list of operation matrices (op_code: first column, op_len: second column) for each query
            record in `df`. If `None` and alignment operations are needed, they are generated from the CIGAR column of
            `df`.
        :param qry_fai: FAI file for query sequences (optional). Needed if features require the query length (i.e.
            proportion of a query in an alignment record).
        """

        try:
            return features.get_features(
                df=df,
                feature_list=self.feature_list,
                score_model=self.score_model,
                op_arr_list=op_arr_list,
                existing_score_model=existing_score_model,
                score_prop_conf=self.score_prop_conf,
                df_qry_fai=qry_fai,
                inplace=False
            )
        except Exception as e:
            raise RuntimeError(f'Failed to get feature table for LCAlignModel {self.name}: {e}')

    def _get_full_qry_len(self, full_qry_len, df, qry_fai):
        """
        For each record in `df`, get the full length of the query sequence (not just the segment of the query sequenced
        aligned in each record). Used to compute the alignment proportion for features.

        :param full_qry_len: If not `None`, return this value instead of recomputing it.
        :param df: Dataframe with 'QRY_ID' column or a Pandas Series of the `QRY_ID` column.
        :param qry_fai: Either a path to the query FASTA FAI index or a Pandas Series with query names as keys and
            full query lengths as values, as returned by `svpoplib.ref.get_df_fai()`. If a path name is given (str), then
            `svpoplib.ref.get_df_fai()` is called to retrieve the query lengths.

        :return: A Pandas Series with full query lengths.
        """

        if full_qry_len is not None:
            return full_qry_len

        if qry_fai is None:
            raise RuntimeError(f'Query FASTA index file is needed to compute features, but is missing for LC model {self.name}')

        if isinstance(qry_fai, str):
            df_fai = svpoplib.ref.get_df_fai(qry_fai)
        else:
            df_fai = qry_fai

        if 'QRY_ID' not in df.columns:
            raise RuntimeError(f'Column "QRY_ID" is needed to compute query length, but is missing for LC model {self.name}')

        return df['QRY_ID'].map(df_fai)

    def check_known_attributes(self):
        """
        Check the model definition for unknown attributes. Raises an exception if any are found. This methods should
        be called by the last step of the model definition process.
        """

        if self.allow_unknown_attributes:
            return

        found_attr = set()

        def find_attr(prefix, subdict):

            prefix_str = f'{prefix}/' if prefix is not None else ''

            for key in subdict.keys():
                found_attr.add(f'{prefix_str}{key}' if prefix is not None else key)
                if isinstance(subdict[key], dict):
                    find_attr(f'{prefix_str}{key}', subdict[key])

        unknown_attr = found_attr - self.known_attr

        if unknown_attr:
            n = len(unknown_attr)
            attr_list = ', '.join(sorted(unknown_attr)[:3]) + ('...' if n > 3 else '')

            raise RuntimeError(f'Found {n} unknown attributes in the LC align model definition "{self.name}": {attr_list}')

    @abc.abstractmethod
    def __call__(self,
                 df: pd.DataFrame,
                 existing_score_model: score.ScoreModel=None,
                 op_arr_list: list[np.ndarray[int, int]]=None,
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
        raise NotImplementedError
