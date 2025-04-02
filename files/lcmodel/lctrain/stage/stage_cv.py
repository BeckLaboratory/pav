"""
Stage: CV (Cross-validation)
"""

import logging

import lctrain.trainer

from .stage import *

import lctrain.util

from .stage_features import StageFeatures

logger = logging.getLogger(__name__)

class StageCV(Stage):
    """Train a model."""

    def __init__(self, lctrain_config: dict, workdir_root: str, outdir: str):
        super().__init__('cv', lctrain_config, workdir_root, outdir)

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

        if run_args is None:
            run_args = dict()
        elif isinstance(run_args, argparse.Namespace):
            run_args = vars(run_args)

        if not run_args.get('cv', False):
            logger.debug('Stage CV: Skipping stage (CV is disabled)')
            return False

        uuid_features_features = None
        uuid_features_split = None

        cv_is_complete = None

        train_params_cached = None

        # Read stage cache
        if os.path.isfile(self.stage_cache_filename):
            with gzip.open(self.stage_cache_filename, 'rt') as f:
                stage_cache = json.load(f)

            uuid_features_features = stage_cache.get('stage_uuid').get('features_features', None)
            uuid_features_split = stage_cache.get('stage_uuid').get('features_split', None)

            train_params_cached = stage_cache.get('train_params', None)

        # Get training config (Ignore data split arguments handled the feature extraction stage)
        train_params = {
            key: val for key, val in self.lctrain_config.items()
                 if key not in {'data'} | StageFeatures.FEATURE_PARAMS | StageFeatures.SPLIT_PARAMS
        }

        # Load feature stage uuids
        stage_version_uuids = self.get_stage_version_uuids('features')

        # Determine if this stage is complete
        if uuid_features_features is None or uuid_features_features != stage_version_uuids.get('features_features', None):
            if uuid_features_features is not None:
                logger.info('Stage CV: Detected mismatch with features stage (features step): Re-running')

            uuid_features_features = stage_version_uuids.get('features_features', None)

            if uuid_features_features is None:
                raise RuntimeError('Missing UUID for features stage (features step) in features cache')

            cv_is_complete = False

        if uuid_features_split is None or uuid_features_split != stage_version_uuids.get('features_split', None):
            if uuid_features_split is not None:
                logger.info('Stage CV: Detected mismatch with features stage (split step): Re-running')

            uuid_features_split = stage_version_uuids.get('features_split', None)

            if uuid_features_split is None:
                raise RuntimeError('Missing UUID for features stage (split step) in features cache')

            cv_is_complete = False

        # Load trainer
        try:
            trainer = lctrain.trainer.get_trainer(
                self, train_params
            )
        except Exception as e:
            raise RuntimeError(f'Error creating trainer: {e}')

        # Determine if this stage is complete
        if cv_is_complete is None:
            if train_params_cached is not None and train_params_cached != train_params:
                logger.info('Stage CV: Detected mismatch with train parameters: Re-running')
                cv_is_complete = False
            else:
                cv_is_complete = trainer.is_cv_complete()

        # Skip stage
        if not force and cv_is_complete:
            logger.debug('Stage CV: All output is current: Skipping stage')
            return False
        elif force:
            logger.debug('Stage CV: Forced execution')
            cv_is_complete = False

        # Run training
        logger.info('Stage CV: Running...')

        if not pretend:
            trainer.train(False, True)

            # Write cache
            cv_cache = {
                'stage_uuid': self.create_stage_uuid(**{
                    'features_features': uuid_features_features,
                    'features_split': uuid_features_split,
                }),
                'train_params': train_params
            }

            os.makedirs(self.workdir, exist_ok=True)

            with gzip.open(self.stage_cache_filename, 'wt') as f:
                json.dump(cv_cache, f, indent=4)

        logger.debug('Stage CV: Complete')
        return True
