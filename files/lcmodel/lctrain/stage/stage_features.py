"""
Stage: Features
"""

import logging
import numpy as np
import pandas as pd
import uuid

import pavlib

import lctrain.const

from .stage import *
from .stage_init import StageInit

logger = logging.getLogger(__name__)

class StageFeatures(Stage):
    """Extract features and create test/train/cv splits."""

    STAGE_FILE_NAMES = {
        'features_filename': 'features_and_labels.npz',
        'split_filename': 'test_cv_sets.npz'
    }

    FEATURE_PARAMS = {
        'score_model',
        'score_prop_conf',
        'score_prop_rescue',
        'score_mm_prop_conf',
        'score_mm_prop_rescue',
        'rescue_length',
        'merge_flank',
        'test_size'  # Note: test_size affects both sets
    }

    SPLIT_PARAMS = {
        'test_size', 'k_fold'
    }

    def __init__(self, lctrain_config: dict, workdir_root: str, outdir: str):
        super().__init__('features', lctrain_config, workdir_root, outdir)

        self.features_filename = self.workdir_path(self.STAGE_FILE_NAMES['features_filename'])
        self.split_filename = self.workdir_path(self.STAGE_FILE_NAMES['split_filename'])

    def run(self,
            force: bool=False,
            pretend: bool=False,
            run_args: argparse.Namespace | dict=None
            ) -> bool:
        """
        Execute this stage.

        :param force: Force all steps of the stage to run.
        :params pretend: Check files and log steps that would be taken, but do not execute.
        :param run_args: Runtime arguments. May be a dict or argparse.Namespace object.

        :return: True if stage was run or would have been run if pretend was not set.
        """

        # Stage state
        uuid_init_train = None

        uuid_features_features = None
        uuid_features_split = None

        feature_params_cached = None
        split_params_cached = None

        features_is_complete = None
        split_is_complete = None

        # Read stage cache
        if os.path.isfile(self.stage_cache_filename):
            with gzip.open(self.stage_cache_filename, 'rt') as f:
                stage_cache = json.load(f)

            uuid_init_train = stage_cache.get('stage_uuid').get('init_train', None)

            uuid_features_features = stage_cache.get('stage_uuid').get('features_features', None)
            uuid_features_split = stage_cache.get('stage_uuid').get('features_split', None)

            feature_params_cached = stage_cache.get('feature_params', None)
            split_params_cached = stage_cache.get('split_params', None)

        # Create feature parameters
        feature_params = {
            key: self.lctrain_config.get(key, getattr(lctrain.const, f'DEFAULT_{key.upper()}'))
                for key in self.FEATURE_PARAMS
        }

        split_params = {
            key: self.lctrain_config.get(key, getattr(lctrain.const, f'DEFAULT_{key.upper()}'))
                for key in self.SPLIT_PARAMS
        }

        # Load init stage uuid
        stage_version_uuids = self.get_stage_version_uuids('init')

        # Determine if this stage is complete
        if uuid_init_train is None or uuid_init_train != stage_version_uuids.get('init_train', None):
            if uuid_init_train is not None:
                logger.info('Stage Features: Detected mismatch with init stage (train step): Re-running')

            uuid_init_train = stage_version_uuids.get('init_train')

            if uuid_init_train is None:
                raise RuntimeError('Missing UUID for init stage (train step) in init cache')

            features_is_complete = False
            split_is_complete = False

        # Check for file completeness
        if features_is_complete is None:
            features_is_complete = self.is_feature_extraction_complete(feature_params, feature_params_cached)

        if split_is_complete is None:
            split_is_complete = self.is_split_complete(split_params, split_params_cached)

        # Check for missing UUIDs on completed steps
        if features_is_complete and uuid_features_features is None:
            raise RuntimeError(f'Missing UUID for complete features stage (features step) in cache filename: {self.stage_cache_filename}')

        if split_is_complete and uuid_features_split is None:
            raise RuntimeError(f'Missing UUID for complete features stage (split step) in cache filename: {self.stage_cache_filename}')

        # Skip stage
        if not force and features_is_complete and split_is_complete:
            logger.debug('Stage Features: All output is current: Skipping stage')
            return False
        elif force:
            logger.debug('Stage Features: Forced execution')
            features_is_complete = False
            split_is_complete = False

        # Create feature table
        if not features_is_complete:
            logger.info('Stage Features: Extracting features')

            if not pretend:
                uuid_features_features = uuid.uuid4().hex

                df_features = self.extract_features(feature_params)

                os.makedirs(self.workdir, exist_ok=True)

                np.savez_compressed(
                    self.features_filename,
                    X=df_features[self.lctrain_config['features']].values,
                    y=df_features['LC'].astype(int).values,
                    datasource_index=df_features['DATASOURCE_INDEX'].values,
                    align_index=df_features['INDEX'].values
                )


        # Create splits
        if not split_is_complete:
            logger.info('Stage Features: Splitting examples into train/test/cv sets')

            if not pretend:
                uuid_features_split = uuid.uuid4().hex

                index_train, index_test, index_crossval_train, index_crossval_val = self.split(split_params)

                # Save
                os.makedirs(self.workdir, exist_ok=True)

                np.savez_compressed(
                    self.split_filename,
                    k_fold=split_params['k_fold'],
                    index_train=index_train,
                    index_test=index_test,
                    **index_crossval_train,
                    **index_crossval_val
                )

        # Update stage cache
        if not pretend:
            with gzip.open(self.stage_cache_filename, 'wt') as f:
                json.dump(
                    {
                        'stage_uuid': self.create_stage_uuid(
                            **{
                                'init_train': uuid_init_train,
                                'features_features': uuid_features_features,
                                'features_split': uuid_features_split
                            }
                        ),
                        'feature_params': feature_params,
                        'split_params': split_params
                    }, f, indent=4
                )

        logger.debug('Stage Features: Complete')
        return True

    def extract_features(self,
                         feature_params: dict
                         ) -> pd.DataFrame:
        """
        Run feature extraction.

        Extract alignment features and produce a table with features (X), labels (y), data source name, and alignment
            index in the data source. Each feature is a column with the feature name, and labels are in the "LC"
            column. The data source name is in the "name" column, and the "INDEX" from the original alignment tables
            are retained in this table. These data make it possible to stratify data for test and cross-validation and
            trace each example back to its source.

        :param feature_params: Feature parameters.

        :return: Feature table augmented with labels and data source information.
        """

        # Read data sources
        df_datasource = pd.read_csv(
            self.workdir_path(StageInit.STAGE_FILE_NAMES['train_datasource_filename'], 'init'),
        )

        # Get score model
        try:
            score_model = pavlib.align.score.get_score_model(feature_params['score_model'])
        except Exception as e:
            raise RuntimeError(f'Error loading score model: {e}')

        # Create lc model
        lc_model = pavlib.align.lcmodel.LCAlignModelNull({
            'type': 'null',
            'features': self.lctrain_config['features'],
            'score_model': score_model,
            'score_prop_conf': feature_params['score_prop_conf']
        })

        # Generate features
        df_features_list = list()

        last_datasource = None

        for index, row_datasource in df_datasource.iterrows():

            if row_datasource['name'] != last_datasource:
                logger.info('Extracting features for data source group (name=%s)', row_datasource['name'])
                last_datasource = row_datasource['name']

            logger.debug('Extracting features for data source (name=%s, assembly=%s, hap=%s)', row_datasource['name'], row_datasource['asm_name'], row_datasource['hap'])

            # Read alignments
            df_align = pd.read_csv(
                row_datasource['align_trim_none'], sep='\t',
                dtype={'#CHROM': str}
            )

            index_set_trim_qry = set(
                pd.read_csv(
                    row_datasource['align_trim_qry'], sep='\t',
                    usecols=['INDEX']
                )['INDEX']
            )

            # Predict features and LC
            df_features = lctrain.features.get_feature_table(df_align, lc_model, row_datasource)

            df_lc = lctrain.features.lc_heuristics(
                df_align, df_features, index_set_trim_qry,
                score_model,
                score_prop_conf=feature_params['score_prop_conf'],
                score_prop_rescue=feature_params['score_prop_rescue'],
                score_mm_prop_conf=feature_params['score_mm_prop_conf'],
                score_mm_prop_rescue=feature_params['score_mm_prop_rescue'],
                rescue_length=feature_params['rescue_length'],
                merge_flank=feature_params['merge_flank']
            )

            df_features = pd.concat([df_features, df_lc], axis=1)

            df_features.insert(0, 'NAME', row_datasource['name'])
            df_features.insert(1, 'DATASOURCE_INDEX', index)
            df_features.insert(2, 'INDEX', df_align['INDEX'])

            df_features_list.append(df_features)

        return pd.concat(df_features_list, axis=0)

    def split(self, split_params: dict) -> tuple[np.ndarray, np.ndarray, dict, dict]:
        """
        Split examples into test and train sets.

        Returns a tuple of 4 items:
        - Train indices
        - Test indices
        - Cross-validation train indices as a dict.
        - Cross-validation validation indices as a dict.

        The cross-validation train and test indices are dictionaries with keys "crossval_train_n" and "crossval_val_n",
        respectively, where n is the cross-validation fold number (from 0 to k-fold - 1).

        :param split_params: Split parameters.

        :return: A tuple of indices.
        """

        import sklearn.model_selection

        # Load features
        feature_loader = np.load(self.features_filename, mmap_mode='r', allow_pickle=False)

        y = feature_loader['y']
        datasource_index = feature_loader['datasource_index']

        del feature_loader

        n = y.shape[0]

        if datasource_index.shape[0] != n:
            raise RuntimeError(f'Error loading features: Number of data sources ({datasource_index.shape[0]}) does not match number of features ({n})')

        # Get the data source group for each data source item
        name_to_index = pd.read_csv(
            self.workdir_path(StageInit.STAGE_FILE_NAMES['train_runpath_filename'], 'init'),
            usecols=['name']
        )['name'].reset_index(drop=False).set_index('name')['index']

        df_name = pd.read_csv(
            self.workdir_path(StageInit.STAGE_FILE_NAMES['train_datasource_filename'], 'init'),
            usecols=['name']
        )

        datasource_group_index = name_to_index[df_name['name']].values

        # Define stratification across LC (values of y) and data source groups (one for each datasource name)
        strata = np.unique(
            np.column_stack((y, datasource_group_index[datasource_index])),  # n x 2 matrix
            return_inverse=True,
            axis=0
        )[1]  # Convert each unique y/LC combination to a unique integer

        # Split train and test sets
        try:
            index_train, index_test = sklearn.model_selection.train_test_split(
                np.arange(n),
                test_size=split_params['test_size'],
                stratify=strata
            )

        except Exception as e:
            raise RuntimeError(f'Error while splitting test and train data: {e}')

        # Create randomized cross-validation splits
        index_crossval = list(
            sklearn.model_selection.StratifiedKFold(
                n_splits=split_params['k_fold'],
                shuffle=True
            ).split(index_train, strata[index_train])
        )

        index_crossval = [
            tuple((
                np.random.permutation(cv) for cv in cv_tuple
            )) for cv_tuple in index_crossval
        ]

        index_crossval_train = {f'crossval_train_{i}': index_crossval[i][0] for i in range(split_params['k_fold'])}
        index_crossval_val = {f'crossval_val_{i}': index_crossval[i][1] for i in range(split_params['k_fold'])}

        # Save
        return index_train, index_test, index_crossval_train, index_crossval_val

    def is_feature_extraction_complete(self, feature_params: dict, feature_params_cached: dict) -> bool:
        """
        Determine if output is complete.

        :return: True if output is complete and False if this step must be re-run.
        """

        if feature_params != feature_params_cached:
            logger.info('Stage Features incomplete: Detected feature configuration changes')
            return False

        # Check for required files
        check_files = [
            self.features_filename
        ]

        missing_files = [filename for filename in check_files if not os.path.exists(filename)]

        if missing_files:
            # Log if partial (don't log on first run)
            if len(missing_files) != len(check_files):
                logger.info('Stage Features incomplete: Missing feature files: %s', ", ".join(missing_files))

            return False

        return True

    def is_split_complete(self, split_params: dict, split_params_cached) -> bool:

        if split_params != split_params_cached:
            logger.info('Stage Features incomplete: Detected train/test/cv split configuration changes')
            return False

        # Check for required files
        check_files = [
            self.split_filename
        ]

        missing_files = [filename for filename in check_files if not os.path.exists(filename)]

        if missing_files:
            # Log if partial (don't log on first run)
            if len(missing_files) != len(check_files):
                logger.info('Stage Features incomplete: Missing split files: %s', ", ".join(missing_files))

            return False

        return True