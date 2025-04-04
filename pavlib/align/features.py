"""
Functions for calculating alignment features.
"""

import numpy as np

import pavlib
import svpoplib

def score(df, score_model):
    """
    Score alignment records.

    :param df: DataFrame of alignment records.
    :param score_model: Alignment score model.

    :return: A Series of alignment record scores.
    """

    if score_model is None:
        raise RuntimeError('Alignment score model is not defined')

    if 'CIGAR' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "CIGAR" column to compute feature "SCORE"')

    return df['CIGAR'].apply(score_model.score_cigar_tuples)

def score_mm(df, score_model):
    """
    Score mismatches in alignment records ignoring gaps.

    :param df: DataFrame of alignment records.
    :param score_model: Alignment score model. A mismatch-only model is derived from this model.

    :return: A Series of alignment record scores.
    """

    if score_model is None:
        raise RuntimeError('Alignment score model is not defined')

    if 'CIGAR' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "CIGAR" column to compute feature "SCORE"')

    score_model_mm = score_model.mismatch_model()

    return df['CIGAR'].apply(score_model_mm.score_cigar_tuples)

def score_prop(df, score_model):
    """
    Alignment score ("SCORE" column) divided by the maximum alignment score (i.e. match(query length)).

    :param df: DataFrame of alignment records.
    :param score_model: Alignment score model.

    :return: A Series of alignment record score proportions.
    """

    if 'SCORE' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "SCORE" column needed to compute the score proportion.')

    if 'QRY_POS' not in df.columns or 'QRY_END' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "QRY_POS" and/or "QRY_END" columns needed to compute the score proportion')

    return df['SCORE'].clip(0.0) / (
            df['QRY_END'] - df['QRY_POS']
    ).apply(score_model.match)

def score_mm_prop(df, score_model):
    """
    Compute the mismatch score proportion as the score determined by matches and mismatches (gaps ignored) divided by
    the number of aligned bases (gaps ignored).

    :param df: DataFrame of alignment records.
    :param score_model: Alignment score model. A mismatch-only model is derived from this model.

    :return: A Series of alignment record score proportions.
    """

    if 'SCORE_MM' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "SCORE_MM" column needed to compute the score proportion.')

    if 'CIGAR' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "CIGAR" column needed to compute the score proportion.')

    score_model_mm = score_model.mismatch_model()

    # Compute proportion
    mm_prop = np.zeros(df.shape[0])
    score_match = df.apply(pavlib.align.util.count_aligned_bases, axis=1).apply(score_model_mm.match)

    can_compute = score_match > 0.0

    mm_prop[can_compute] = df['SCORE_MM'].values[can_compute] / score_match[can_compute]

    return mm_prop

def mismatch_prop(df):
    """
    Compute the proportion of mismatches alignment records.

    The mismatch proportion is defined as the number of mismatches divided by the number of aligned bases. Unlike
    the mismatch score proportion, this mismatch proportion is not based on an alignment score model where matches and
    mismatches are typically weighted differently.

    :param df: DataFrame of alignment records.

    :return: A Series of alignment record mismatch proportions.
    """

    cigar_tuples = df['CIGAR'].apply(pavlib.align.util.cigar_str_to_tuples)



    df['CIGAR'].apply(svpoplib.align.util.count_mismatches)


def anchor_proportion(df, score_prop_conf):
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

    return np.where(
        df['SCORE_PROP'] >= score_prop_conf,
        2.0,
        (df['QRY_ORDER'].values >= anchor_min) * 1.0 + (df['QRY_ORDER'].values <= anchor_max) * 1.0
    ) / 2.0

def query_proportion(df, df_qry_fai):
    if 'QRY_POS' not in df.columns or 'QRY_END' not in df.columns or 'QRY_ID' not in df.columns:
        raise RuntimeError(f'Alignment DataFrame is missing the "QRY_ID", "QRY_POS", and/or "QRY_END" column(s) to compute the query proportion')

    if df_qry_fai is None:
        raise RuntimeError(f'Missing "df_qry_fai" parameter needed to compute compute the query proportion')

    if isinstance(df_qry_fai, str):
        df_qry_fai = svpoplib.ref.get_df_fai(df_qry_fai)

    return (df['QRY_END'] - df['QRY_POS']) / df['QRY_ID'].map(df_qry_fai)
