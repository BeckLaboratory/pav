"""
Stage: Init
"""

import logging
import uuid

import pavlib

import lctrain.util

from .stage import *

logger = logging.getLogger(__name__)

class StageInit(Stage):
    """Locate data sources."""

    STAGE_FILE_NAMES = {
        'train_runpath_filename': 'runpath_train.csv.gz',
        'train_datasource_filename': 'datasource_train.csv.gz',
        'eval_runpath_filename': 'eval_runpath.csv.gz',
        'eval_datasource_filename': 'eval_datasource.csv.gz'
    }

    KEY_TRAIN_SET = {
        'train_runpath_filename',
        'train_datasource_filename',
    }

    KEY_EVAL_SET = set(STAGE_FILE_NAMES.keys()) - KEY_TRAIN_SET

    def __init__(self, lctrain_config: dict, workdir_root: str, outdir: str):
        super().__init__('init', lctrain_config, workdir_root, outdir)

        self.stage_file_names = {
            key: self.workdir_path(value) for key, value in self.STAGE_FILE_NAMES.items()
        }

        self.train_runpath_filename = self.stage_file_names['train_runpath_filename']
        self.train_datasource_filename = self.stage_file_names['train_datasource_filename']
        self.eval_runpath_filename = self.stage_file_names['eval_runpath_filename']
        self.eval_datasource_filename = self.stage_file_names['eval_datasource_filename']

        self.check_files_train = [self.stage_file_names[key] for key in self.KEY_TRAIN_SET]
        self.check_files_eval = [self.stage_file_names[key] for key in self.KEY_EVAL_SET]

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

        data_config = self.lctrain_config['data']

        # Read stage cache
        uuid_init_eval = None
        uuid_init_train = None

        data_config_cached = None

        train_is_complete = None
        eval_is_complete = None

        if os.path.isfile(self.stage_cache_filename):
            with gzip.open(self.stage_cache_filename, 'rt') as f:
                stage_cache = json.load(f)

            uuid_init_eval = stage_cache.get('stage_uuid').get('init_eval', None)
            uuid_init_train = stage_cache.get('stage_uuid').get('init_train', None)
            data_config_cached = stage_cache.get('data', None)

        # Check if this stage is complete
        if train_is_complete is None:
            train_is_complete = self.is_train_complete(data_config, data_config_cached)

        if not train_is_complete:
            eval_is_complete = False  # Force eval stage if train was not complete
        else:
            if eval_is_complete is None:
                eval_is_complete = self.is_eval_complete(data_config, data_config_cached)

        # Check for defined UUIDs
        if train_is_complete and uuid_init_train is None:
            raise RuntimeError(f'Missing UUID for complete init stage (train step) in cache filename: {self.stage_cache_filename}')

        if eval_is_complete and uuid_init_eval is None:
            raise RuntimeError(f'Missing UUID for complete init stage (all steps) in cache filename: {self.stage_cache_filename}')

        # Stop if stages are complete
        if not force and train_is_complete and eval_is_complete:
            logger.debug('Stage Init: All output is current: Skipping stage')
            return False
        elif force:
            logger.debug('Stage Init: Forced execution')
            train_is_complete = False
            eval_is_complete = False

        # Load data sources
        logger.debug('Stage Init: Loading data sources')

        df_runpath = None
        df_datasource = None

        if not pretend:
            # Resolve data sources
            df_runpath, df_datasource = lctrain.util.load_data_sources(data_config)

        # Save training sources
        if not train_is_complete:
            logger.debug('Stage Init: Saving training data sources')

            if not pretend:
                uuid_init_train = uuid.uuid4().hex
                os.makedirs(self.workdir, exist_ok=True)

                os.makedirs(self.workdir, exist_ok=True)
                df_runpath.loc[df_runpath['train']].to_csv(self.train_runpath_filename, index=False, compression='gzip')
                df_datasource.loc[df_datasource['train']].to_csv(self.train_datasource_filename, index=False, compression='gzip')

        # Save eval sources
        if not eval_is_complete:
            logger.debug('Stage Init: Saving eval data sources')

            if not pretend:
                uuid_init_eval = uuid.uuid4().hex

                os.makedirs(self.workdir, exist_ok=True)
                df_runpath.to_csv(self.eval_runpath_filename, index=False, compression='gzip')
                df_datasource.to_csv(self.eval_datasource_filename, index=False, compression='gzip')

        # Save init cache
        logger.debug('Stage Init: Saving init cache')

        if not pretend:
            os.makedirs(self.workdir, exist_ok=True)

            with gzip.open(self.stage_cache_filename, 'wt') as f:
                json.dump(
                    {
                        'data': data_config,
                        'stage_uuid': self.create_stage_uuid(
                            **{
                                'init_eval': uuid_init_eval,
                                'init_train': uuid_init_train
                            }
                        )
                    },
                    f, indent=4
                )

        logger.debug('Stage Init: Complete')
        return True

    def is_train_complete(self,
                    data_config: list[dict] | None,
                    data_config_cached: list[dict] | None
                    ) -> bool:
        """
        Determine if training data sources are complete.

        :return: True if step "train" is complete.
        """

        #
        # Check training set
        # Note: Comments in the data config are ignored for training (not for eval)
        #

        # Check for training configuration file
        if data_config_cached is None:
            logger.debug('Stage Init incomplete: No previous config')
            return False

        # Check config
        if lctrain.util.data_config_strip(data_config, True) != lctrain.util.data_config_strip(data_config_cached, True):
            logger.info('Stage Init incomplete: Detected changes to training and eval data sources')
            return False

        # Check for required files for training
        missing_files_train = [filename for filename in self.check_files_train if not os.path.exists(filename)]

        if missing_files_train:
            logger.info('Stage Init incomplete: Missing files required for training and eval: %s', ", ".join(missing_files_train))
            return False

        return True


    def is_eval_complete(self,
                    data_config: list[dict] | None,
                    data_config_cached: list[dict] | None
                    ) -> bool:
        """
        Determine if eval data sources are complete..

        :return: True if step "eval" is complete.
        """

        # Check for training configuration file
        if data_config != data_config_cached:
            logger.info('Stage Init partially incomplete: Detected changes to eval data sources')
            return False

        # Check for required files for training
        missing_files_eval = [filename for filename in self.check_files_eval if not os.path.exists(filename)]

        if missing_files_eval:
            logger.info('Stage Init partially incomplete: Missing files: %s', ", ".join(missing_files_eval))
            return False

        return True
