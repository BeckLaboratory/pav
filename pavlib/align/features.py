"""
Functions for calculating alignment features.
"""

import numpy as np
import pandas as pd

from . import score
from . import op

import svpoplib

# Features PAV saves to alignment BED files
ALIGN_FEATURE_COLUMNS = ['SCORE', 'SCORE_PROP', 'MATCH_PROP', 'QRY_PROP']

KNOWN_FEATURE_SET = {
    'SCORE', 'SCORE_PROP',
    'SCORE_MM', 'SCORE_MM_PROP',
    'MATCH_PROP', 'ANCHOR_PROP', 'QRY_PROP'
}

# Use this value if ANCHOR_PROP is in the alignment features stored in BED files.
ALIGN_FEATURE_SCORE_PROP_CONF = 0.85

def get_features(
        df: pd.DataFrame | pd.Series,
        feature_list: list[str]=None,
        score_model: score.ScoreModel=None,
        existing_score_model: str | score.ScoreModel | bool=None,
        score_prop_conf: float=ALIGN_FEATURE_SCORE_PROP_CONF,
        op_arr_list: list[np.ndarray[int, int]]=None,
        df_qry_fai: pd.Series=None,
        inplace: bool=False,
        only_features: bool=True,
        force_all: bool=False
) -> pd.DataFrame | pd.Series:
    """
    Get a table of alignment features.

    :param df: DataFrame or Series of alignment records. If a Series is given, then it is converted to a single-row
        DataFrame.
    :param feature_list: Features to compute. If None, recompute any features in `df`.
    :param score_model: Model to use to compute alignment scores. If None and a feature requires a score model, an
        exception is raised. May be a string specification for a score model.
    :param existing_score_model: The score model used to compute features already in `df`. If this is not None and
        `score_model` is not compatible with it, then scores are recomputed. If None, assume existing scores do not
        need to be recomputed. As a special case, a boolean value of `True` forces re-computation, and `False` skips it.
        May be a string specification for a score model.
    :param score_prop_conf: When determining if an alignment is anchored by a confident alignment upstream or
        downstream, this is the minimum score proportion to flag an alignment as confident.
    :param op_arr_list: If set, this is a list of alignment operation arrays for each record where each array has two
        columns (op_code and op_len). If not set and a feature uses alignment operations, then it is extracted from the
        CIGAR string for each record, but this can be expensive over many alignments.
    :param df_qry_fai: FAI file for query sequences (optional). Needed if features require the query length (i.e.
        proportion of a query in an alignment record).
    :param inplace: If True, modify `df` in place. The table is still returned by the function whether or not it is
        altered in place avoiding a copy but allowing chaining logic.
    :param only_features: If True, only return alignment features, else, return the whole table.
    :param force_all: If True, compute all features, even if they are already present.

    :return: DataFrame or Series (matching the type of `df`) of alignment features.
    """

    is_series = False

    if score_prop_conf is None:
        score_prop_conf = ALIGN_FEATURE_SCORE_PROP_CONF

    # Check DataFrame
    if isinstance(df, pd.Series):
        df = pd.DataFrame([df])
        is_series = True

    elif isinstance(df, pd.DataFrame):
        if not inplace:
            df = df.copy()
    else:
        raise RuntimeError(f'Alignment DataFrame is not a Pandas DataFrame or Series: {type(df)}')

    # Get score models
    if existing_score_model is not None and not isinstance(existing_score_model, bool):
        if isinstance(existing_score_model, str):
            existing_score_model = score.get_score_model(existing_score_model)

    if score_model is not None:
        if isinstance(score_model, str):
            score_model = score.get_score_model(score_model)

    if score_model is not None and existing_score_model is not None:
        if isinstance(existing_score_model, bool):
            rescore = existing_score_model
        else:
            rescore = score_model != existing_score_model
    else:
        rescore = True

    if force_all:
        rescore = True

    # Check for unknown features
    if feature_list is not None:
        feature_set = set(feature_list)
    else:
        feature_set = set(df.columns) & KNOWN_FEATURE_SET

    unknown_features = feature_set - KNOWN_FEATURE_SET

    if unknown_features:
        n = len(unknown_features)
        s = ', '.join(sorted(unknown_features)[:3]) + (f'...' if n > 3 else '')
        raise RuntimeError(f'Found {n:,d} unknown features: {s}')

    # Compute or update features
    df = df.copy()

    if feature_set & {'SCORE',  'SCORE_PROP', 'ANCHOR_PROP'}:

        if rescore or 'SCORE' not in df.columns:
            op_arr_list = _set_op_array(df, op_arr_list)

            try:
                df['SCORE'] = feature_score(df, score_model, op_arr_list)
            except Exception as e:
                raise RuntimeError(f'Failed to compute alignment scores ("SCORE"): {e}')

    if 'SCORE_PROP' in feature_set:
        if rescore or 'SCORE_PROP' not in df.columns:
            try:
                df['SCORE_PROP'] = feature_score_prop(df, score_model)
            except Exception as e:
                raise RuntimeError(f'Failed to compute alignment score proportions ("SCORE_PROP"): {e}')

    if feature_set & {'SCORE_MM', 'SCORE_MM_PROP'}:
        if rescore or 'SCORE_MM' not in df.columns:
            op_arr_list = _set_op_array(df, op_arr_list)

            try:
                df['SCORE_MM'] = feature_score_mm(df, score_model, op_arr_list)
            except Exception as e:
                raise RuntimeError(f'Failed to compute alignment scores ("SCORE_MM"): {e}')

    if 'SCORE_MM_PROP' in feature_set:
        if rescore or 'SCORE_MM_PROP' not in df.columns:
            op_arr_list = _set_op_array(df, op_arr_list)

            try:
                df['SCORE_MM_PROP'] = feature_score_mm_prop(df, score_model, op_arr_list)
            except Exception as e:
                raise RuntimeError(f'Failed to compute alignment score proportions ("SCORE_MM_PROP"): {e}')

    if 'MATCH_PROP' in feature_set:

        if force_all or 'MATCH_PROP' not in df.columns:
            op_arr_list = _set_op_array(df, op_arr_list)

            try:
                df['MATCH_PROP'] = feature_match_prop(df, op_arr_list)
            except Exception as e:
                raise RuntimeError(f'Failed to compute alignment match proportions ("MATCH_PROP"): {e}')

    if 'ANCHOR_PROP' in feature_set:
        if score_prop_conf is None:
            raise RuntimeError(f'Missing "score_prop_conf" for computing alignment anchor proportions ("ANCHOR_PROP")')

        if force_all or 'ANCHOR_PROP' not in df.columns:
            try:
                df['ANCHOR_PROP'] = feature_anchor_prop(df, score_prop_conf)
            except Exception as e:
                raise RuntimeError(f'Failed to compute alignment anchor proportions ("ANCHOR_PROP"): {e}')

    if 'QRY_PROP' in feature_set:
        if force_all or 'QRY_PROP' not in df.columns:
            try:
                df['QRY_PROP'] = feature_qry_prop(df, df_qry_fai)
            except Exception as e:
                raise RuntimeError(f'Failed to compute alignment query proportions ("QRY_PROP"): {e}')

    # Trim to just features
    if only_features:
        df = df[feature_list]

    # Return feature table
    if is_series:
        return df.iloc[0]

    return df

def feature_score(
        df: pd.DataFrame,
        score_model: score.ScoreModel,
        op_arr_list: list[np.ndarray[int, int]]=None
) -> pd.Series:
    """
    Score alignment records.

    If `df` is a DataFrame, the "CIGAR" column is used from that DataFrame. If it is a Series, then it is assumed to be
    the CIGAR column af an alignment DataFrame. For each CIGAR string, a list of CIGAR tuples is derived from the
    "CIGAR" column and used for scoring.

    if `df` is a list, then it is a list of CIGAR tuples for each alignment record already extracted from the CIGAR
    column of the alignment record. `df` must then be a list (one element per alignment record) of lists (list of cigar
    tuples) where each tuple is two integers (opcode, oplen).

    Several features need the list of CIGAR tuples for each alignment record. For these, it is significantly faster to
    compute the list of tuples once and pass it to each feature rather than extract the CIGAR tuples from the CIGAR
    string multiple times.

    :param df: DataFrame of alignment records including a "CIGAR" column.
    :param score_model: Alignment score model.
    :param op_arr_list: List of alignment operation (op_code: first column, op_len: second column), one array
        per alignment record.

    :return: A Series of alignment record scores.
    """

    if score_model is None:
        raise RuntimeError('Alignment score model is not defined')

    op_arr_list = _set_op_array(df, op_arr_list)

    return pd.Series(
        [
            score_model.score_operations(op_arr) for op_arr in op_arr_list
        ],
        index=df.index
    )

def feature_score_mm(
        df: pd.DataFrame,
        score_model: score.ScoreModel,
        op_arr_list: list[np.ndarray[int, int]]=None
) -> pd.Series:
    """
    Score aligned bases in alignment records ignoring gaps. This uses a match/mismatch-only alignment score model
        derived from `score_model.mismatch_model()`.

    :param df: DataFrame of alignment records including a "CIGAR" column.
    :param score_model: Alignment score model.
    :param op_arr_list: List of alignment operation (op_code: first column, op_len: second column), one array
        per alignment record.

    :return: A Series of alignment record scores.
    """

    if score_model is None:
        raise RuntimeError('Alignment score model is not defined')

    return feature_score(df, score_model.mismatch_model(), op_arr_list)

def feature_score_prop(
        df: pd.DataFrame,
        score_model: score.ScoreModel
) -> pd.Series:
    """
    Divide the alignment score ("SCORE" column) by the maximum alignment score if all query bases were aligned and
    matched (i.e. score / (match_score * query_length)).

    :param df: DataFrame of alignment records.
    :param score_model: Alignment score model.

    :return: A Series of alignment record score proportions.
    """

    if 'SCORE' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "SCORE" column needed to compute the score proportion.')

    if 'QRY_POS' not in df.columns or 'QRY_END' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "QRY_POS" and/or "QRY_END" columns needed to compute the score proportion')

    if score_model is None:
        raise RuntimeError('Alignment score model is not defined')

    return (
        df['SCORE'].clip(0.0) / (
            df['QRY_END'] - df['QRY_POS']
        ).apply(score_model.match)
    ).fillna(0.0)

def feature_score_mm_prop(
        df: pd.DataFrame,
        score_model: score.ScoreModel,
        op_arr_list: list[np.ndarray[int, int]]=None
) -> pd.Series:
    """
    Divide the mismatch alignment score ("SCORE_MM" column, gap penalties ignored) by the maximum alignment score if all
    non-gap query bases were aligned and matched (i.e. score / (match_score * aligned_bases)).

    :param df: DataFrame of alignment records.
    :param score_model: Alignment score model.
    :param op_arr_list: List of alignment operation (op_code: first column, op_len: second column), one array
        per alignment record.

    :return: A Series of alignment record score proportions.
    """

    if 'SCORE_MM' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "SCORE_MM" column needed to compute the mismatch score proportion.')

    if score_model is None:
        raise RuntimeError('Alignment score model is not defined')

    op_arr_list = _set_op_array(df, op_arr_list)

    return pd.Series(
        np.nan_to_num(
            df['SCORE_MM'].values / np.array(
                [
                    (
                        op_arr[:, 1] * ((op_arr[:, 0] == op.EQ) + (op_arr[:, 0] == op.X))
                    ).sum() for op_arr in op_arr_list
                ]
            ), nan=0.0
        ),
        index=df.index
    )

def feature_match_prop(
        df: pd.DataFrame,
        op_arr_list: list[np.ndarray[int, int]]=None
) -> pd.Series:
    """
    Compute the proportion of matching bases over aligned bases (i.e. EQ / (X + EQ)).

    The match proportion is defined as the number of matches divided by the number of aligned bases. Unlike
    the match score proportion, this match proportion is not based on an alignment score model where matches and
    matches are typically weighted differently.

    :param df: DataFrame of alignment records.
    :param op_arr_list: List of alignment operation (op_code: first column, op_len: second column), one array
        per alignment record.

    :return: A Series of alignment record match proportions.
    """

    op_arr_list = _set_op_array(df, op_arr_list)

    # Note: Match rate is EQ / (X + EQ) ; M not counted (should have been checked already)
    # EQ := Number of matches (CIGAR op "=")
    # X  := Number of mismatches (CIGAR op "X")
    #
    # Direct computation summing EQ and X once:
    # EQ / (X + EQ)   =   1 / (1 + X/EQ)
    return pd.Series(
        [
            1 / (
                1 + np.nan_to_num(
                    (
                        op_arr[:, 1] * (op_arr[:, 0] == op.X)
                    ).sum() / (
                        op_arr[:, 1] * (op_arr[:, 0] == op.EQ)
                    ).sum(),
                    nan=0.0
                )
            )
                for op_arr in op_arr_list
        ],
        index=df.index
    )

def feature_anchor_prop(
        df: pd.DataFrame,
        score_prop_conf: float
) -> pd.Series:
    """
    Determine if an alignment record is between high-confidence alignment records along the query sequence. Values
    are reported as 0.0 (no confident alignment on the query sequence), 0.5 (at least confident alignment upstream or
    downstream, but not both), and 1.0 (at least one confident alignment both upstream and downstream).

    :param df: DataFrame of alignment records.
    :param score_prop_conf: Minimum score proportion for a confident alignment record.

    :return: A Series of alignment record anchor proportions.
    """

    missing_cols = {'QRY_ID', 'QRY_ORDER'} - set(df.columns)

    if missing_cols:
        raise RuntimeError(f'Missing column(s) to compute feature ANCHOR_PROP: {", ".join(sorted(missing_cols))}')

    qry_order_gr = df.loc[df['SCORE_PROP'] >= score_prop_conf].groupby('QRY_ID')['QRY_ORDER']

    anchor_min = df['QRY_ID'].map(
        qry_order_gr.min().reindex(df['QRY_ID'].unique()).astype(float).fillna(np.inf)
    )

    anchor_max = df['QRY_ID'].map(
        qry_order_gr.max().reindex(df['QRY_ID'].unique()).astype(float).fillna(-np.inf)
    )

    return pd.Series(
        np.where(
            df['SCORE_PROP'] >= score_prop_conf,
            2.0,
            (df['QRY_ORDER'].values >= anchor_min) * 1.0 + (df['QRY_ORDER'].values <= anchor_max) * 1.0
        ) / 2.0,
        index=df.index
    )

def feature_qry_prop(
        df: pd.DataFrame,
        df_qry_fai: pd.Series
) -> pd.Series:
    """
    Get the proportion of the query sequence aligned in this record.

    :param df: DataFrame of alignment records.
    :param df_qry_fai: FAI file for query sequences (key: sequence name, value: sequence length).

    :return: A Series of alignment record query proportions.
    """
    if 'QRY_POS' not in df.columns or 'QRY_END' not in df.columns or 'QRY_ID' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "QRY_ID", "QRY_POS", and/or "QRY_END" column(s) to compute the query proportion')

    if df_qry_fai is None:
        raise RuntimeError(f'Missing "df_qry_fai" parameter needed to compute compute the query proportion')

    if isinstance(df_qry_fai, str):
        df_qry_fai = svpoplib.ref.get_df_fai(df_qry_fai)

    return (df['QRY_END'] - df['QRY_POS']) / df['QRY_ID'].map(df_qry_fai)

def _set_op_array(
        df: pd.DataFrame,
        op_arr_list: list[np.ndarray[int, int]]=None,
) -> list[np.ndarray[int, int]]:
    """
    Get a list of alignment operation tuples per alignment record.

    :param df: DataFrame of alignment records.
    :param op_arr_list: List of alignment operation (op_code: first column, op_len: second column), one array
        per alignment record.
    """
    if op_arr_list is not None:
        return op_arr_list

    if 'CIGAR' not in df.columns:
        raise RuntimeError(f'Cannot retrieve alignment operations from alignment table: Missing "CIGAR" column')

    return [
        op.cigar_as_array(cigar) for cigar in df['CIGAR']
    ]
