"""Calculate alignment features.

Alignment features are derived from alignment tables and provide summary statistics for alignment records.
"""

__all__ = [
    'ALIGN_FEATURE_SCORE_PROP_CONF',
    'ALIGN_TABLE_COLUMNS',
    'FeatureGenerator',
    'feature',
]

from dataclasses import dataclass, field
from functools import wraps
import inspect
from typing import Any, Callable, Iterable, Optional

import polars as pl

from .score import ScoreModel, get_score_model

from . import op

# Features PAV saves to alignment tables
ALIGN_TABLE_COLUMNS = (
    'score', 'score_prop', 'match_prop', 'qry_prop'
)

# Use this value if ANCHOR_PROP is in the alignment features stored in BED files.
ALIGN_FEATURE_SCORE_PROP_CONF = 0.85

_feature_registry: dict[str, Callable] = {}
_feature_schema: dict[str, type[pl.DataType]] = {}


def feature(dtype: type[pl.DataType], name: str = None) -> Callable:
    """Register a feature function to FeatureGenerator (decorator).

    The decorator creates a wrapper that accepts standard parameters but only passes the parameters that the decorated
    function actually accepts.

    :param dtype: Data type of feature.
    :param name: Name of feature. If None, uses the function name.

    :returns: The decorated function with registration side effects.
    """

    def decorator(func: Callable) -> Callable:
        nonlocal name

        if name is None:
            name = func.__name__

        # Get function signature
        param_names = set(inspect.signature(func).parameters.keys()) - {'self'}

        @wraps(func)
        def wrapper(
                self,
                df: pl.DataFrame,
                df_qry_fai: Optional[pl.DataFrame] = None,
                temp_features: Optional[set[str]] = None
        ) -> pl.DataFrame:
            """Wrap feature functions with a consistent signature.

            Only passes parameters that the decorated function accepts.

            :param self: The FeatureGenerator instance.
            :param df: Table of alignment records.
            :param df_qry_fai: Table of sequence lengths (optional).
            :param temp_features: Set to track temporary features (optional).

            :returns: A table with new or updated feature columns.
            """
            # Build kwargs with only the parameters the function accepts
            kwargs: dict[str, Any] = {'df': df}

            if 'df_qry_fai' in param_names and df_qry_fai is not None:
                kwargs['df_qry_fai'] = df_qry_fai

            if 'temp_features' in param_names and temp_features is not None:
                kwargs['temp_features'] = temp_features

            return (
                func(self, **kwargs)
                .with_columns(pl.col(name).cast(dtype))
            )

        # Register the wrapper in the class registry
        # This will be done when the class is defined
        # if not hasattr(FeatureGenerator, '_feature_registry'):
        #     FeatureGenerator._feature_registry = {}

        _feature_registry[name] = wrapper
        _feature_schema[name] = dtype

        def wrapper_no_if(
                self, *args, **kwargs
        ):
            """Wrap function and add dtype cast.

            :param self: The FeatureGenerator instance.
            :param args: Positional arguments.
            :param kwargs: Keyword arguments.

            :returns: A table with new or updated feature columns.
            """
            return (
                func(self, *args, **kwargs)
                .with_columns(pl.col(name).cast(dtype))
            )

        return wrapper_no_if

    return decorator


@dataclass(frozen=True)
class FeatureGenerator:
    """Class for generating alignment features.

    :param features: Names of features to compute. If empty or None, re-compute existing features..
    :param score_model: Model to use to compute alignment scores. A string specification is used to construct a score
        model (the object attribute is always a ScoreModel object).
    :param score_prop_conf: When determining if an alignment is anchored by a confident alignment upstream or
        downstream, this is the minimum score proportion to flag an alignment as confident.
    :param only_features: If True, only return alignment features, else, return the full alignment table.
    :param force_all: If True, compute all features, even if they are already present.
    :param force_score: If True, recompute alignment scores even if they are already present in the alignment table.
    """

    features: Optional[Iterable[str] | str] = field(default=ALIGN_TABLE_COLUMNS)
    score_model: ScoreModel | str = field(default_factory=get_score_model)
    score_prop_conf: float = field(default=ALIGN_FEATURE_SCORE_PROP_CONF)
    only_features: bool = field(default=False)
    force_all: bool = field(default=False)
    force_score: bool = field(default=False)

    # # Class-level feature registry
    # _feature_registry: dict[str, Callable] = field(init=False, repr=False, default_factory=dict)
    # _feature_schema: dict[str, type[pl.DataType]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self):
        """Post-initialization."""
        object.__setattr__(self, 'features', self.__class__.get_feature_list(self.features))

        if isinstance(self.score_model, str):
            object.__setattr__(self, 'score_model', get_score_model(self.score_model))

        if self.score_model is None:
            object.__setattr__(self, 'score_model', get_score_model())

        # Validate parameter types
        if not isinstance(self.score_prop_conf, (int, float)):
            raise ValueError(f'score_prop_conf must be numeric: {type(self.score_prop_conf)}')

        if not isinstance(self.only_features, bool):
            raise ValueError(f'only_features must be boolean: {type(self.only_features)}')

        if not isinstance(self.force_all, bool):
            raise ValueError(f'force_all must be boolean: {type(self.force_all)}')

        if not isinstance(self.force_score, bool):
            raise ValueError(f'force_score must be boolean: {type(self.force_score)}')

    def __call__(
            self,
            df: pl.DataFrame,
            df_qry_fai: Optional[pl.DataFrame] = None
    ) -> pl.DataFrame:
        """Get a table of alignment features.

        :param df: DataFrame of alignment records.
        :param df_qry_fai: FAI file for query sequences. Needed if features require the query length (i.e. proportion of
            a query in an alignment record).

        :returns: A DataFrame of alignment features.

        :raises ValueError: If the requested features could not be generated. Features might be unknown, or the
            alignment table is missing required columns.
        """
        # Set features to compute
        auto_features = False  # True if features were extracted from df columns

        if not self.features:
            features = list(df.columns)
            auto_features = True
        else:
            features = self.features

        # Compute features
        found_features = set()
        temp_features = set()

        for feature in features:

            if (
                feature in df.columns and
                not self.force_all and
                not (
                    self.force_score and feature in {'score', 'score_mm'}
                )
            ):
                found_features.add(feature)
                continue

            try:
                df = self.append_feature(feature, df, df_qry_fai, temp_features)
                found_features.add(feature)

            except Exception as e:
                raise ValueError(f'Failed to generate feature "{feature}": {e}') from e

        # Report errors
        if not auto_features and (unknown_features := set(features) - found_features):
            n = len(unknown_features)
            s = ', '.join(sorted(unknown_features)[:3]) + ('...' if n > 3 else '')
            raise ValueError(f'Could not generate {n:,d} unknown features: {s}')

        # Only return features
        if self.only_features:
            df = df.select(features)

        # Remove temporary features
        df = df.drop(temp_features - found_features, strict=False)

        # Return dataframe
        return df

    def append_feature(
            self,
            feature_name: str,
            df: pl.DataFrame,
            df_qry_fai: Optional[pl.DataFrame],
            temp_features: Optional[set[str]],
            is_temp: bool = False
    ) -> pl.DataFrame:
        """Append a feature to a DataFrame.

        :param feature_name: Name of the feature.
        :param df: Table of alignment records.
        :param df_qry_fai: Table of sequence lengths.
        :param temp_features: Set to track temporary features. Features marked as temporary will be added to this set.
        :param is_temp: If True, add feature name to temp_features.

        :returns: A DataFrame with a new or updated feature column.

        :raises ValueError: If the feature is unknown.
        """
        if feature_name not in _feature_registry:
            raise ValueError(f'Unknown feature: {feature_name}')

        df = _feature_registry[feature_name](self, df, df_qry_fai, temp_features)

        if is_temp:
            if temp_features is None:
                raise ValueError('Cannot mark feature as temp: temp_features is None')
            temp_features.add(feature_name)

        return df

    def get_schema(self) -> dict[str, type[pl.DataType]]:
        """Get a schema for features in this feature generator.

        :returns: A dictionary mapping feature names to feature data types.
        """
        return {
            feature: _feature_schema[feature]
            for feature in self.features
        }

    @classmethod
    def all_feature_schema(cls) -> dict[str, type[pl.DataType]]:
        """Get a schema for all features.

        :returns: A dictionary mapping feature names to feature data types.
        """
        return _feature_schema.copy()

    @feature(pl.Float32)
    def score(
            self,
            df: pl.DataFrame
    ) -> pl.DataFrame:
        """Score alignment records.

        :param df: Table of alignment records.

        :returns: A table with new or updated feature columns.
        """
        return df.with_columns(
            self.score_model.score_align_table(df)
            .alias('score')
        )

    @feature(pl.Float32)
    def score_mm(
            self,
            df: pl.DataFrame
    ) -> pl.DataFrame:
        """Score aligned bases in alignment records.

        Score ignores gaps by using `score_model.mismatch_model()`.

        :param df: Table of alignment records.

        :returns: A table with new or updated feature columns.
        """
        return df.with_columns(
            self.score_model.mismatch_model().score_align_table(df)
            .alias('score_mm')
        )

    @feature(pl.Float32)
    def score_prop(
            self,
            df: pl.DataFrame,
            df_qry_fai: pl.DataFrame,
            temp_features: set[str]
    ) -> pl.DataFrame:
        """Score proportion.

        Divide the alignment score ("SCORE" column) by the maximum alignment score if all query bases were aligned and
        matched (i.e. score / (match_score * query_length)).

        :param df: Table of alignment records.
        :param df_qry_fai: Table of sequence lengths.
        :param temp_features: Temporary features generated by this function are added to this set.

        :returns: A table with new or updated feature columns.
        """
        if 'score' not in df.columns:
            df = self.append_feature('score', df, df_qry_fai, temp_features, is_temp=True)

        return df.with_columns(
            df
            .select(
                pl.col('score') / (
                    (
                        pl.col('qry_end') - pl.col('qry_pos')
                    )
                    .fill_null(0.0)
                    .map_elements(self.score_model.match, return_dtype=pl.Float32)
                )
            )
            .to_series()
            .alias('score_prop')
        )

    @feature(pl.Float32)
    def score_mm_prop(
            self,
            df: pl.DataFrame,
            df_qry_fai: pl.DataFrame,
            temp_features: set[str]
    ) -> pl.DataFrame:
        """
        Score mismatch proportion.

        Divide the mismatch alignment score ("score_mm" column, gap penalties ignored) by the maximum alignment score if
        all non-gap query bases were aligned and matched (i.e. score / (match_score * aligned_bases)).

        :param df: Table of alignment records.
        :param df_qry_fai: Table of sequence lengths.
        :param temp_features: Temporary features generated by this function are added to this set.

        :returns: A table with new or updated feature columns.
        """
        if 'score_mm' not in df.columns:
            df = self.append_feature('score_mm', df, df_qry_fai, temp_features, is_temp=True)

        return (
            df.lazy()
            .with_columns(
                pl.col('align_ops').struct.field("op_code").alias('_op_code'),
                pl.col('align_ops').struct.field("op_len").alias('_op_len')
            )
            .with_columns(
                (
                    pl.col('score_mm') / (
                        (
                            pl.col("_op_len") * (
                                pl.col("_op_code").list.eval(pl.element().is_in([op.EQ, op.X]))
                            )
                        )
                        .list.sum()
                        .map_elements(self.score_model.match, return_dtype=pl.Float32)
                    )
                )
                .alias('score_mm_prop')
            )
            .drop(['_op_code', '_op_len'])
            .collect()
        )

    @feature(pl.Float32)
    def match_prop(
            self,
            df: pl.DataFrame
    ) -> pl.DataFrame:
        """Score proportion of maximum.

        Compute the proportion of matching bases over aligned bases (i.e. EQ / (X + EQ)).

        The match proportion is defined as the number of matches divided by the number of aligned bases. Unlike
        the match score proportion, this match proportion is not based on an alignment score model where matches and
        mismatches are typically weighted differently.

        :param df: Table of alignment records.

        :returns: A table with new or updated feature columns.
        """
        return (
            df.lazy()
            .with_columns(
                pl.col('align_ops').struct.field("op_code").alias('_op_code'),
                pl.col('align_ops').struct.field("op_len").alias('_op_len')
            )
            .with_columns(
                (
                    (
                        (
                            pl.col("_op_len") * (
                                pl.col("_op_code").list.eval(pl.element() == op.EQ)
                            )
                        )
                        .list.sum()
                    ) / (
                        (
                            pl.col("_op_len") * (
                                pl.col("_op_code").list.eval(pl.element().is_in([op.EQ, op.X]))
                            )
                        )
                        .list.sum()
                    )
                ).alias('match_prop')
            )
            .drop(['_op_code', '_op_len'])
            .collect()
        )

    @feature(pl.Float32)
    def anchor_prop(
            self,
            df: pl.DataFrame
    ) -> pl.DataFrame:
        """Anchor proportion.

        Determine if an alignment record is between high-confidence alignment records along the query sequence. Values
        are reported as 0.0 (no confident alignment on the query sequence), 0.5 (at least confident alignment upstream
        or downstream, but not both), and 1.0 (at least one confident alignment both upstream and downstream).

        :param df: Table of alignment records.

        :returns: A table with new or updated feature columns.
        """
        df_qry_min_max = (
            df
            .filter(
                pl.col('score_prop') >= self.score_prop_conf
            )
            .group_by('qry_id')
            .agg(
                qry_min=(
                    pl.col('qry_order').min()
                ),
                qry_max=(
                    pl.col('qry_order').max()
                )
            )
        )

        return df.with_columns(
            df.join(
                df_qry_min_max,
                on='qry_id',
                how='left'
            )
            .select(
                (
                    (
                        (
                            (pl.col('qry_order') > pl.col('qry_min')).fill_null(False)
                        ) + (
                            (pl.col('qry_order') < pl.col('qry_max')).fill_null(False)
                        )
                    ) / 2.0
                )
            )
            .to_series()
            .alias('anchor_prop')
        )

    @feature(pl.Float32)
    def qry_prop(
            self,
            df: pl.DataFrame,
            df_qry_fai: pl.DataFrame
    ) -> pl.DataFrame:
        """Get the proportion of the query sequence aligned in this record.

        :param df: Table of alignment records.
        :param df_qry_fai: Table of sequence lengths.

        :returns: A table with new or updated feature columns.
        """
        return df.with_columns(
            df
            .join(
                df_qry_fai.rename({'chrom': 'qry_id', 'len': 'qry_len'}, strict=False),
                on='qry_id', how='left'
            )
            .select(
                (pl.col('qry_end') - pl.col('qry_pos')) / pl.col('qry_len')
            )
            .to_series()
            .alias('qry_prop')
        )

    @classmethod
    def get_feature_list(
            cls,
            features: Iterable[str] | str | None
    ) -> tuple[str, ...]:
        """Get a list of features from a feature argument.

        If features is an iterable, it is converted to a tuple. If features is a string and matches a feature keyword,
        then a pre-configured feature tuple is returned. If None, then the alignment table default columns are returned.

        Feature keywords:
            all: All known features.
            align: Default columns of a PAV alignment table.

        :param features: Features to compute.

        :returns: List of feature names.
        """
        if features is None:
            return ALIGN_TABLE_COLUMNS

        elif isinstance(features, str):
            if features == 'all':
                features = cls.all_features()
            elif features == 'align':
                features = ALIGN_TABLE_COLUMNS
            else:
                features = tuple(features.split(','))

        try:
            features = tuple(features)
        except TypeError as e:
            raise ValueError(f'Unrecognized feature type, expected string or iterable: {type(features)}') from e

        if unknown_features := set(features) - set(cls.all_features()):
            raise ValueError(f'Unrecognized feature(s): {", ".join(sorted(unknown_features))}')

        return features

    @staticmethod
    def all_features() -> tuple[str, ...]:
        """Get all available features.

        :returns: List of feature names.
        """
        return tuple(_feature_registry.keys())
