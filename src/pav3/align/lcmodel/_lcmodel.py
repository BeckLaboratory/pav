"""A model for flagging low-confidence (LC) alignment records.

Example model JSON::

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

__all__ = [
    'LCAlignModel',
]

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pathlib
import importlib.resources.abc
import inspect

import frozendict
import numpy as np
import polars as pl
from typing import Any, Optional

from ...io import ResourceReader

from ..score import ScoreModel, get_score_model
from ..features import ALIGN_FEATURE_SCORE_PROP_CONF, FeatureGenerator
from ...util import as_bool


@dataclass(frozen=True, repr=False)
class LCAlignModel(ABC):
    """Base class for low-confidence alignment models.

    Warning: Child class implementations must call `super().__post_init__()`.

    :ivar lc_model_def: LC model definition dictionary from a model definition JSON file. If None, the child class
        must create one or let this parent class raise an exception. Null models do this, most others will not.
    :ivar model_dir: Directory containing the model definition JSON file. If `None`, no related resources can be loaded
        and the full model definition must be in `lc_model_def`.
    """

    lc_model_def: Optional[dict[str, Any]]
    resource_type: Optional[str]
    anchor: Optional[str]

    def __post_init__(self):
        """Check model invariants."""
        if self.lc_model_def is None:
            raise ValueError('Model definition is missing')

        # Copy model definition
        lc_model_def = dict(self.lc_model_def)

        # Check and set
        if lc_model_def.get('type', None) is None:
            raise ValueError('LC align model definition is missing the "type" attribute')

        lc_model_def['type'] = str(lc_model_def['type']).strip()

        if lc_model_def['type'].strip() == '':
            raise ValueError('LC align model definition has an empty "type" attribute')

        lc_model_def['type_version'] = int(lc_model_def.get('type_version', 0))

        lc_model_def['score_model'] = get_score_model(self.lc_model_def.get('score_model', None))

        lc_model_def['features'] = FeatureGenerator.get_feature_list(lc_model_def.get('features', []))

        lc_model_def['score_prop_conf'] = float(
            self.lc_model_def.get('score_prop_conf', ALIGN_FEATURE_SCORE_PROP_CONF)
        )

        lc_model_def['name'] = str(lc_model_def.get('name', '<MODEL_NAME_NOT_SPECIFIED>'))

        lc_model_def['description'] = (
            str(lc_model_def.get('description')).strip()
            if 'description' in lc_model_def else None
        )

        lc_model_def['allow_unknown_attributes'] = as_bool(
            self.lc_model_def.get('allow_unknown_attributes', False)
        )

        # Freeze attributes
        object.__setattr__(self, 'lc_model_def', frozendict.frozendict(lc_model_def))

    @property
    def type(self) -> str:
        """LC align model type."""
        return self.lc_model_def['type']

    @property
    def type_version(self) -> int:
        """Version for the type. May guide the behavior of the class implementing the model (defaults to 0)."""
        return self.lc_model_def['type_version']

    @property
    def score_model(self) -> ScoreModel:
        """Alignment scoring model."""
        return self.lc_model_def['score_model']

    @property
    def features(self) -> tuple[str, ...]:
        """Alignment features used by the model."""
        return self.lc_model_def['features']

    @property
    def score_prop_conf(self) -> float:
        """Alignment score proportion confidence threshold for determining low-confidence alignment records."""
        return self.lc_model_def['score_prop_conf']

    @property
    def name(self) -> str:
        """LC align model name."""
        return self.lc_model_def['name']

    @property
    def description(self) -> Optional[str]:
        """LC align model description, if defined (otherwise None)."""
        return self.lc_model_def['description']

    @property
    def allow_unknown_attributes(self) -> bool:
        """Whether to allow unknown attributes in the alignment table (defaults to False)."""
        return self.lc_model_def['allow_unknown_attributes']

    def __repr__(self) -> str:
        """Get a string representation."""
        return f'{self.__class__.__name__}(name={self.name}, type={self.type}, type_version={self.type_version})'

    def resource_path(
            self,
            resource_name: str
    ) -> pathlib.Path | importlib.resources.abc.Traversable:
        """Get a path to a resource relative to the model directory.

        :param resource_name: Name of the resource relative to the model directory.

        :returns: Path to the resource.

        :raises ValueError: If the model directory was not specified when the model class was instantiated.
        """

        if self.resource_type == 'package':
            return importlib.resources.files(self.anchor) / resource_name

        elif self.resource_type == 'filesystem':
            return pathlib.Path(self.anchor) / resource_name

        raise ValueError(f'Unknown resource type: {self.resource_type}')

    def resource_reader(
            self,
            resource_name: str,
            text_mode: bool = True
    ) -> ResourceReader:
        """Get a reader for a resource relative to the model directory.

        :param resource_name: Name of the resource relative to the model directory.
        :param text_mode: Whether to open the resource in text mode (defaults to True).

        :returns: Reader for the resource.

        :raises ValueError: If the model directory was not specified when the model class was instantiated.
        """
        return ResourceReader(self.anchor, resource_name, self.resource_type, text_mode)

    def get_feature_table(
            self,
            df: pl.DataFrame,
            existing_score_model: Optional[str | ScoreModel | bool] = None,
            df_qry_fai: pl.DataFrame = None,
            only_features: bool = True
    ) -> pl.DataFrame:
        """Get a table of alignment features for a model.

        Scores for alignment records in `df` were created using tunable score model parameters, which may or may not
        match the score model that the LC model was trained on. Since recomputing scores requires nontrivial time, these
        scores are recomputed only if necessary.

        Existing features are recomputed only if necessary. For features requiring a score model, if both `score_model`
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
            `score_model` is not compatible with it, then scores are recomputed. If None, assume existing scores
            do not need to be recomputed. As a special case, a boolean value of `True` forces re-computation, and
            `False` skips it. Value may be a string specification for a score model.
        :param df_qry_fai: FAI file for query sequences (optional). Needed if features require the query length (i.e.
            proportion of a query in an alignment record).
        :param only_features: If True, only features are returned. If False, the full alignment table with features is
            returned.

        :returns: A DataFrame of alignment features.
        """
        force_score = self.score_model != get_score_model(existing_score_model)

        try:
            return FeatureGenerator(
                features=self.features,
                score_model=self.score_model,
                only_features=only_features,
                force_score=force_score
            )(
                df, df_qry_fai
            )

        except Exception as e:
            raise ValueError(f'Failed to get feature table for LCAlignModel {self.name}: {e}') from e

    @classmethod
    def get_properties(cls) -> set[str]:
        """Get named properties of a class."""
        return {
            name for name, obj in
            inspect.getmembers(cls, lambda a: isinstance(a, property))
        }

    def check_unknown_attributes(self):
        """Check the model definition for unknown attributes.

        Raises an exception if any are found. This methods should be called by the last step of the model definition.

        :raises ValueError: Unknown attributes are found.
        """
        if self.allow_unknown_attributes:
            return

        unknown_attr = self.lc_model_def.keys() - self.get_properties()

        if unknown_attr:
            n = len(unknown_attr)
            attr_list = ', '.join(sorted(unknown_attr)[:3]) + ('...' if n > 3 else '')

            raise ValueError(
                f'Found {n} unknown attributes in the LC align model definition "{self.name}": {attr_list}'
            )

    @abstractmethod
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
        pass
