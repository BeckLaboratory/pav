"""
Base class and definitions for model trainers.
"""

import abc
import gzip
import json
import logging
import os

import lctrain.stage
import lctrain.trainer

logger = logging.getLogger(__name__)

def get_trainer(
        stage: lctrain.stage.Stage,
        train_config: dict
):

    model_name_camel = train_config['type']
    model_name_camel = model_name_camel[0].upper() + model_name_camel[1:].lower()

    model_class = getattr(lctrain.trainer, f'ModelTrainer{model_name_camel}')
    return model_class(stage, train_config)

class ModelTrainer(object, metaclass=abc.ABCMeta):
    """
    Model trainer base class.
    """

    def __init__(self,
                 model_type: str,
                 stage: lctrain.stage.Stage,
                 train_config: dict,
                 model_files: dict=None,
                 accessory_files_train: dict=None,
                 accessory_files_cv: dict=None
                 ):

        """
        Create a model trainer.

        :param name: Model name.
        :param model_type: Model type.
        :param stage: Stage this trainer is executed on.
        :param train_config: Training configuration.
        :param model_files: Files that should be output to the model's output directory. This directory contains all
            the files PAV needs to run the model.
        :param accessory_files: Files output by the training process, but do not belong in the model's output
            directory. These files can include training history for each epoch and other training artifacts.
        """

        # Check
        if model_type is not None:
            model_type = str(model_type).strip()

        if not model_type:
            raise ValueError('Model type is empty')

        if stage is None:
            raise ValueError('Stage must be set')

        if train_config is None:
            raise ValueError('Train configuration must be set')

        if model_files is None:
            model_files = dict()

        if accessory_files_train is None:
            accessory_files = dict()

        if accessory_files_cv is None:
            accessory_files = dict()

        # Set attributes
        self.model_type = model_type
        self.stage = stage
        self.train_config = train_config

        self.name = self.stage.lctrain_config['name']

        self.model_files = {
            key: os.path.join(self.stage.outdir, filename) for key, filename in model_files.items()
        }

        self.accessory_files_train = {
            key: self.stage.workdir_path(filename) for key, filename in accessory_files_train.items()
        }

        self.accessory_files_cv = {
            key: self.stage.workdir_path(filename) for key, filename in accessory_files_cv.items()
        }

        self.features_filename = self.stage.workdir_path(lctrain.stage.StageFeatures.STAGE_FILE_NAMES['features_filename'], 'features')
        self.split_filename = self.stage.workdir_path(lctrain.stage.StageFeatures.STAGE_FILE_NAMES['split_filename'], 'features')
        self.features_cache_filename = self.stage.workdir_path(lctrain.stage.Stage.STAGE_CACHE_FILENAME_BASE, 'features')

    @abc.abstractmethod
    def train(self, run_train=True, run_cv=True) -> None:
        """
        Train this model.

        :param run_train: Run training step.
        :param run_cv: Run cross-validation step.
        """
        raise NotImplementedError

    def is_training_complete(self) -> bool:

        # Check for required files
        check_files = list(self.model_files.values()) + list(self.accessory_files_train.values())

        missing_files = [filename for filename in check_files if not os.path.exists(filename)]

        if missing_files:
            # Log if partial (don't log on first run)
            if len(missing_files) != len(check_files):
                logger.info(
                    'Training incomplete for model %s: Missing feature files: %s',
                    self.name,
                    ', '.join(missing_files[:3]) + ('...' if len(missing_files) > 3 else '')
                )

            return False

        return True

    def is_cv_complete(self) -> bool:

        # Check for required files
        check_files = list(self.accessory_files_cv.values())

        missing_files = [filename for filename in check_files if not os.path.exists(filename)]

        if missing_files:
            # Log if partial (don't log on first run)
            if len(missing_files) != len(check_files):
                logger.info(
                    'Cross-validation incomplete for model %s: Missing feature files: %s',
                    self.name,
                    ', '.join(missing_files[:3]) + ('...' if len(missing_files) > 3 else '')
                )

            return False

        return True
