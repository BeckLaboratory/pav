"""
Construct features for training.
"""

import collections
import intervaltree
import pandas as pd

import pavlib
import svpoplib

import lctrain.const

def get_feature_table(df_align, lc_model, row_datasource):

    # Get features
    return lc_model.get_feature_table(
        df=df_align,
        existing_score_model=get_existing_score_model_name(row_datasource),
        qry_fai=svpoplib.ref.get_df_fai(row_datasource['fai_path'])
    )

def get_existing_score_model_name(row_datasource):
    """
    Get the name of an existing score model.
    """

    return pd.read_csv(
        row_datasource['align_stats'], sep='\t', header=None, names=['key', 'val'], index_col='key'
    )['val']['SCORE_MODEL']


def lc_heuristics(
        df_align, df_features, index_set_trim_qry,
        score_model,
        score_prop_conf, score_mm_prop_conf,
        score_prop_rescue, score_mm_prop_rescue,
        rescue_length, merge_flank
):
    """
    Get non-model predictions for low-confidence alignments. These predictions become the feature labels for training.

    :param df_align: Alignment table.
    :param df_features: Feature table.
    :param index_set_trim_qry: Set of alignment indices for which query sequences were not completely removed by trimming.
    :param score_model: Alignment score model for this LC model.
    :param score_prop_conf: Alignment score threshold for low-confidence alignments.
    :param score_mm_prop_conf: Alignment score mismatch threshold for low-confidence alignments.
    :param score_prop_rescue: Alignment score threshold for rescuing  alignments from clustering.
    :param score_mm_prop_rescue: Alignment score mismatch threshold for rescuing alignments from clustering.
    :param rescue_length: Query length threshold for rescuing low-confidence alignments. Must match this and one of
        `score_prop_rescue` or `score_mm_prop_rescue` to avoid being marked as LC when it clusters with other LC
        records.
    :param merge_flank: Alignment length threshold for merging low-confidence alignments while gathering other alignment
        records that cluster with them.
    """

    # Get score columns if needed
    if score_prop_conf is not None or score_prop_rescue is not None:
        if 'SCORE_PROP' not in df_features.columns:
            if 'SCORE' not in df_align.columns:
                df_align = df_align.copy()
                df_align['SCORE'] = pavlib.align.features.score(df_align, score_model)

            score_prop_col = pavlib.align.features.score_prop(df_align, score_model)

        else:
            score_prop_col = df_features['SCORE_PROP']
    else:
        score_prop_col = None

    if score_mm_prop_conf is not None or score_mm_prop_rescue is not None:
        if 'SCORE_MM_PROP' not in df_features.columns:
            if 'SCORE_MM' not in df_align.columns:
                df_align = df_align.copy()
                df_align['SCORE_MM'] = pavlib.align.features.score_mm(df_align, score_model)

            score_prop_mm_col = pavlib.align.features.score_mm_prop(df_align, score_model)

        else:
            score_prop_mm_col = df_features['SCORE_MM_PROP']
    else:
        score_prop_mm_col = None

    # Get query length
    qry_len = df_align['QRY_END'] - df_align['QRY_POS']

    # LC by alignment score
    lc_score = pd.Series([False] * df_align.shape[0])
    lc_score.name = 'LC_SCORE'

    if score_prop_conf is None and score_mm_prop_conf is None:
        raise RuntimeError('One or both of "score_prop_conf" and "score_prop_mm_conf" parameters must be defined.')

    if score_prop_conf is not None:
        if 'SCORE_PROP' not in df_features.columns:
            raise RuntimeError('score_prop_conf cannot be used without feature "SCORE_PROP"')

        lc_score |= score_prop_col < score_prop_conf

    if score_mm_prop_conf is not None:
        if 'SCORE_MM_PROP' not in df_features.columns:
            raise RuntimeError('score_prop_mm_conf cannot be used without feature "SCORE_MM_PROP"')

        lc_score |= score_prop_mm_col < score_mm_prop_conf

    # LC by trimming (was completely removed)
    lc_trim = ~ df_align['INDEX'].isin(index_set_trim_qry)
    lc_trim.name = 'LC_TRIM'

    # Create an interval tree with low-confidence regions (merge overlapping regions)
    itree = collections.defaultdict(intervaltree.IntervalTree)

    for index, row in df_align.loc[lc_trim | lc_score].iterrows():
        pos = row['POS']
        end = row['END']

        for tree_match in itree[row['#CHROM']][row['POS'] - merge_flank:row['END'] + merge_flank]:
            pos = min([pos, tree_match.begin])
            end = max([end, tree_match.end])

            itree[row['#CHROM']].remove(tree_match)

        itree[row['#CHROM']][pos:end] = True

    # Mark rescued alignments (for region association)
    if rescue_length is not None:
        rescue_length = lctrain.const.DEFAULT_RESCUE_LENGTH

    lc_rescue = pd.Series([False] * df_align.shape[0])

    if score_prop_rescue is not None:
        lc_rescue |= (score_prop_col >= score_prop_rescue) & (qry_len >= rescue_length)

    if score_mm_prop_rescue is not None:
        lc_rescue |= (score_prop_mm_col >= score_mm_prop_rescue) & (qry_len >= rescue_length)

    # LC by region association
    lc_region = pd.Series([False] * df_align.shape[0])
    lc_region.name = 'LC_REGION'

    for index, row in df_align.loc[~(lc_trim | lc_score | lc_rescue)].iterrows():
        for tree_match in itree[row['#CHROM']][row['POS']:row['END']]:
            if tree_match.begin <= row['POS'] and tree_match.end >= row['END']:
                lc_region[index] = True

    lc_any = lc_trim | lc_score | lc_region
    lc_any.name = 'LC'

    return pd.concat([lc_trim, lc_score, lc_region, lc_any], axis=1)
