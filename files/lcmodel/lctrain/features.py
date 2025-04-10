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


def get_features_and_lc_heuristics(
        row_datasource: pd.Series,
        score_model: pavlib.align.score.ScoreModel,
        feature_params: dict
) -> pd.DataFrame:
    """
    Get non-model predictions for low-confidence alignments. These predictions become the feature labels for training.

    :param row_datasource: Data source row.
    :param score_model: Alignment score model for this LC model.
    :param feature_params: Feature parameters.

    :return: Table with label columns "LC" (bool) indicating predicted low-confidence alignments based on parameters in
        feature_params. Additional boolean columns "LC_TRIM", "LC_SCORE", "LC_REGION" report which heuristics set
        LC (LC is set for a record if any of these are True).
    """

    # Set parameters
    score_prop_conf = feature_params.get('score_prop_conf', None)
    score_prop_rescue = feature_params.get('score_prop_rescue', None)
    score_mm_prop_conf = feature_params.get('score_mm_prop_conf', None)
    score_mm_prop_rescue = feature_params.get('score_mm_prop_rescue', None)
    match_prop_conf = feature_params.get('match_prop_conf', None)
    match_prop_rescue = feature_params.get('match_prop_rescue', None)
    rescue_length = feature_params.get('rescue_length', None)
    merge_flank = feature_params.get('merge_flank', None)

    # Read
    df = pd.read_csv(
        row_datasource['align_trim_none'], sep='\t',
        dtype={'#CHROM': str}
    )

    index_set_trim_qry = set(
        pd.read_csv(
            row_datasource['align_trim_qry'], sep='\t',
            usecols=['INDEX']
        )['INDEX']
    )

    df_qry_fai = svpoplib.ref.get_df_fai(row_datasource['fai_path'])

    # Determine features needed for heuristics (may be additional to features needed for training)
    lc_feature_set = set()

    if score_prop_conf is not None or score_prop_rescue is not None:
        lc_feature_set.add('SCORE_PROP')

    if score_mm_prop_conf is not None or score_mm_prop_rescue is not None:
        lc_feature_set.add('SCORE_MM_PROP')

    if match_prop_conf is not None or match_prop_rescue is not None:
        lc_feature_set.add('MATCH_PROP')

    feature_list_lc = feature_params['features'] + list(lc_feature_set - set(feature_params['features']))

    # Add features
    df = pavlib.align.features.get_features(
        df=df,
        feature_list=feature_list_lc,
        score_model=score_model,
        existing_score_model=get_existing_score_model_name(row_datasource),
        score_prop_conf=score_prop_conf,
        op_arr_list=None,
        qry_fai=df_qry_fai,
        inplace=True,
        only_features=False,
        force_all=False
    )

    # df_features = lctrain.features.get_feature_table(df_align, lc_model, row_datasource)

    # Get score columns if needed
    score_prop_col = None
    score_prop_mm_col = None
    match_prop_col = None

    if score_prop_conf is not None or score_prop_rescue is not None:
        score_prop_col = df['SCORE_PROP']

    if score_mm_prop_conf is not None or score_mm_prop_rescue is not None:
        score_prop_mm_col = df['SCORE_MM_PROP']

    if match_prop_conf is not None or match_prop_rescue is not None:
        match_prop_col = df['MATCH_PROP']

    # Get query length
    qry_len = df['QRY_END'] - df['QRY_POS']

    # LC by alignment score
    lc_score = pd.Series([False] * df.shape[0])
    lc_score.name = 'LC_SCORE'

    if score_prop_conf is None and score_mm_prop_conf is None and match_prop_conf is None:
        raise RuntimeError('At least one of "score_prop_conf", "match_prop_conf", or "score_prop_mm_conf" parameters must be defined.')

    if score_prop_conf is not None:
        lc_score |= score_prop_col < score_prop_conf

    if score_mm_prop_conf is not None:
        lc_score |= score_prop_mm_col < score_mm_prop_conf

    if match_prop_conf is not None:
        lc_score |= match_prop_col < match_prop_conf

    # LC by trimming (was completely removed)
    lc_trim = ~ df['INDEX'].isin(index_set_trim_qry)
    lc_trim.name = 'LC_TRIM'

    # Create an interval tree with low-confidence regions (merge overlapping regions)
    itree = collections.defaultdict(intervaltree.IntervalTree)

    for index, row in df.loc[lc_trim | lc_score].iterrows():
        pos = row['POS']
        end = row['END']

        for tree_match in itree[row['#CHROM']][row['POS'] - merge_flank:row['END'] + merge_flank]:
            pos = min([pos, tree_match.begin])
            end = max([end, tree_match.end])

            itree[row['#CHROM']].remove(tree_match)

        itree[row['#CHROM']][pos:end] = True

    # Mark rescued alignments (for region association)
    if rescue_length is None:
        rescue_length = lctrain.const.DEFAULT_RESCUE_LENGTH

    lc_rescue = pd.Series([False] * df.shape[0])

    if score_prop_rescue is not None:
        lc_rescue |= (score_prop_col >= score_prop_rescue) & (qry_len >= rescue_length)

    if score_mm_prop_rescue is not None:
        lc_rescue |= (score_prop_mm_col >= score_mm_prop_rescue) & (qry_len >= rescue_length)

    if match_prop_rescue is not None:
        lc_rescue |= (match_prop_col >= match_prop_rescue) & (qry_len >= rescue_length)

    lc_rescue.name = 'LC_RESCUE'

    # LC by region association
    lc_region = pd.Series([False] * df.shape[0])
    lc_region.name = 'LC_REGION'

    for index, row in df.loc[~(lc_trim | lc_score | lc_rescue)].iterrows():
        for tree_match in itree[row['#CHROM']][row['POS']:row['END']]:
            if tree_match.begin <= row['POS'] and tree_match.end >= row['END']:
                lc_region[index] = True

    lc_any = lc_trim | lc_score | lc_region
    lc_any.name = 'LC'

    return pd.concat([df[['INDEX'] + feature_params['features']], lc_trim, lc_score, lc_region, lc_any], axis=1)
