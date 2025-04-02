"""
Logistic regression trainer.
"""

import gzip
import json
import os
import scipy.special
import logging
import numpy as np

import lctrain.const
import lctrain.stage

import pavlib

from . import trainer

logger = logging.getLogger(__name__)

class ModelTrainerLogistic(trainer.ModelTrainer):
    """
    Logistic regression trainer.
    """

    DEFAULT_THRESHOLD = 0.5
    DEFAULT_EPOCHS = 16
    DEFAULT_BATCH_SIZE = 128

    TYPE_VERSION = 0

    MODEL_FILES = {
        'weights': 'weights.npz',
        'model_def': 'model.json'
    }

    ACC_FILES_CV = {
        'cv_history': 'cv_history.json.gz',
        'cv_roc': 'cv_roc.json.gz',
        'cv_history_png': 'cv_history.png',
        'cv_history_svg': 'cv_history.svg',
        'cv_roc_png': 'cv_roc.png',
        'cv_roc_svg': 'cv_roc.svg',
        'model_weights': 'model_weights_per_fold.npz',
        'model_def': 'model_def.json'
    }

    ACC_FILES_TRAIN = {
        'train_history': 'train_history.json.gz',
        'train_roc': 'train_roc.json.gz',
        'train_history_png': 'train_history.png',
        'train_history_svg': 'train_history.svg',
        'train_roc_png': 'train_roc.png',
        'train_roc_svg': 'train_roc.svg'
    }

    # Training configuration parameters to remove before saving the model definition
    PRIVATE_TRAIN_PARAMS = {'epochs', 'batch_size', 'crossval'}

    def __init__(self,
                 stage: lctrain.stage.Stage,
                 train_config: dict
                 ):

        super().__init__(
            model_type='logistic',
            stage=stage,
            train_config=train_config,
            model_files=self.MODEL_FILES.copy(),
            accessory_files_train=self.ACC_FILES_TRAIN.copy(),
            accessory_files_cv=self.ACC_FILES_CV.copy()
        )

        # Check and/or set model version
        if 'type_version' in self.train_config and self.train_config['type_version'] != self.TYPE_VERSION:
            raise RuntimeError(f'Unsupported model version set in the training configuration: {self.train_config["type_version"]}')
        else:
            self.train_config['type_version'] = self.TYPE_VERSION

    def train(self, run_train=True, run_cv=True) -> None:

        # Check parameters
        known_params = {
            'name', 'description',
            'type', 'type_version',
            'features', 'threshold',
            'epochs', 'batch_size', 'crossval'
        }

        unknown_keys = set(self.train_config.keys()) - known_params

        if unknown_keys:
            raise RuntimeError(f'Unknown training parameters: {", ".join(unknown_keys)}')

        # Set defaults
        threshold = float(self.train_config.get('threshold', self.DEFAULT_THRESHOLD))
        epochs = int(self.train_config.get('epochs', self.DEFAULT_EPOCHS))
        batch_size = int(self.train_config.get('batch_size', self.DEFAULT_BATCH_SIZE))

        # Save back to train_config (output as part of the model's JSON)
        self.train_config['threshold'] = threshold

        # Create model
        logger.debug('Loading TensorFlow...')
        import tensorflow as tf
        logger.debug('Loading TensorFlow complete')

        # Load data
        data_loader_features = np.load(self.features_filename, allow_pickle=False)

        X = data_loader_features['X']
        y = data_loader_features['y']

        del data_loader_features

        data_loader_split = np.load(self.split_filename, allow_pickle=False)

        index_train = data_loader_split['index_train']
        index_test = data_loader_split['index_test']

        k_fold = data_loader_split['k_fold']

        index_crossval_train = [
            data_loader_split[f'crossval_train_{i}'] for i in range(k_fold)
        ]

        index_crossval_val = [
            data_loader_split[f'crossval_val_{i}'] for i in range(k_fold)
        ]

        del data_loader_split

        # Run cross-validation
        if run_cv:
            self.run_cv(
                   X, y,
                   index_crossval_train,
                   index_crossval_val,
                   k_fold, epochs, batch_size
            )

        # Run training
        if run_train:
            model = self.run_train(
                X, y,
                index_train,
                index_test,
                epochs, batch_size
            )

            # Save model
            os.makedirs(self.stage.outdir, exist_ok=True)

            model_weights = {
                'w': model.get_weights()[0],
                'b': model.get_weights()[1]
            }

            np.savez_compressed(self.model_files['weights'], **model_weights)

            # Create model definition
            model_def = {
                key: val for key, val in self.train_config.items()
                    if key not in self.PRIVATE_TRAIN_PARAMS
            }

            # Add score model definition
            with gzip.open(self.features_cache_filename, 'rt') as f:
                feature_params = json.load(f).get('feature_params', {})

                if 'score_model' in feature_params:
                    model_def['score_model'] = feature_params['score_model']

                if 'score_prop_conf' in feature_params:
                    model_def['score_prop_conf'] = feature_params['score_prop_conf']

            # Save model definition
            with open(self.model_files['model_def'], 'wt') as f:
                json.dump(model_def, f, indent=4)

    def run_cv(self,
               X, y,
               index_crossval_train,
               index_crossval_val,
               k_fold, epochs, batch_size
               ):
        """
        Run cross-validation.

        :param X: Feature matrix.
        :param y: Labels.
        :param index_crossval_train: List of training indices.
        :param index_crossval_val: List of validation indices.
        :param k_fold: Number of cross-validation folds.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        """

        train_history_list = list()
        y_pred_list = list()

        model_weights = dict()

        model_def = {
            key: val for key, val in self.train_config.items()
                if key not in self.PRIVATE_TRAIN_PARAMS
        }

        for k in range(k_fold):
            X_train = X[index_crossval_train[k]]
            y_train = y[index_crossval_train[k]]
            X_test = X[index_crossval_val[k]]
            y_test = y[index_crossval_val[k]]

            logger.info(f'Training fold {k+1}/{k_fold}...')

            model, train_history = self.train_model(
                X_train, y_train,
                X_test, y_test,
                epochs=epochs,
                batch_size=batch_size
            )

            model_weights.update({
                f'w_{k}': model.get_weights()[0],
                f'b_{k}': model.get_weights()[1]
            })

            train_history_list.append(train_history.history)

            # Get predictions
            logger.debug(f'Getting predictions for fold {k+1}/{k_fold}...')
            y_pred_list.append(
                scipy.special.expit(
                    model.predict(X_test)
                )
            )

        # Write model QC information
        logger.info(f'Evaluating CV')

        fig_train = lctrain.fig.get_training_fig(
            train_history_list,
            type_train='train', type_val='validation'
        )

        fig_roc, roc_auc_list = lctrain.fig.get_roc_fig(
            [
                y[index_crossval_val[k]] for k in range(k_fold)
            ],
            y_pred_list
        )

        # Save CV results
        os.makedirs(self.stage.workdir, exist_ok=True)

        with gzip.open(self.accessory_files_cv['cv_history'], 'wt') as f:
            f.write(json.dumps(train_history_list))

        with gzip.open(self.accessory_files_cv['cv_roc'], 'wt') as f:
            f.write(json.dumps(roc_auc_list))

        fig_train.savefig(self.accessory_files_cv['cv_history_png'])
        fig_train.savefig(self.accessory_files_cv['cv_history_svg'])
        fig_roc.savefig(self.accessory_files_cv['cv_roc_png'])
        fig_roc.savefig(self.accessory_files_cv['cv_roc_svg'])

        np.savez_compressed(
            self.accessory_files_cv['model_weights'],
            **model_weights
        )

        with open(self.accessory_files_cv['model_def'], 'wt') as f:
            json.dump(model_def, f, indent=4)

    def run_train(self,
                  X, y,
                  index_train,
                  index_test,
                  epochs, batch_size
                  ):
        """
        Run training on all training data.

        :param X: Feature matrix.
        :param y: Labels.
        :param index_train: List of training indices.
        :param index_test: List of testing indices.
        :param k_fold: Number of cross-validation folds.
        :param epochs: Number of epochs.
        :param batch_size: Batch size.
        """

        train_history_list = list()
        y_pred_list = list()

        X_train = X[index_train]
        y_train = y[index_train]
        X_test = X[index_test]
        y_test = y[index_test]

        logger.info(f'Training full model...')

        model, train_history = self.train_model(
            X_train, y_train,
            X_test, y_test,
            epochs=epochs,
            batch_size=batch_size
        )

        train_history_list.append(train_history.history)

        # Get predictions
        logger.debug(f'Getting predictions for model...')
        y_pred_list.append(
            scipy.special.expit(  # Logistic function, model outputs logits
                model.predict(X_test)
            )
        )

        # Write model QC information
        logger.info(f'Evaluating model')

        fig_train = lctrain.fig.get_training_fig(
            train_history_list,
            type_train='train', type_val='test'
        )

        fig_roc, roc_auc_list = lctrain.fig.get_roc_fig(
            [
                y_test
            ],
            y_pred_list
        )

        # Save training results
        os.makedirs(self.stage.workdir, exist_ok=True)

        with gzip.open(self.accessory_files_train['train_history'], 'wt') as f:
            f.write(json.dumps(train_history_list))

        with gzip.open(self.accessory_files_train['train_roc'], 'wt') as f:
            f.write(json.dumps(roc_auc_list))

        fig_train.savefig(self.accessory_files_train['train_history_png'])
        fig_train.savefig(self.accessory_files_train['train_history_svg'])
        fig_roc.savefig(self.accessory_files_train['train_roc_png'])
        fig_roc.savefig(self.accessory_files_train['train_roc_svg'])

        return model

    def train_model(self,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    epochs: int=DEFAULT_EPOCHS,
                    batch_size: int=DEFAULT_BATCH_SIZE,
                    ):
        """
        Train a model.

        :param X_train: Training features
        :param y_train: Training labels
        :param X_test: Test features
        :param y_test: Test labels

        :return: A tuple of the model and training history returned by model.fit().
        """

        import tensorflow as tf

        # Small models like these take more time on GPUs from moving parameters around.
        with lctrain.util.tf_cpu():
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input((X_train.shape[1],), name='Input'),
                tf.keras.layers.Dense(1, activation='linear', name='Output')
            ])

            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(
                    from_logits=True
                ),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy']
            )

            train_history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
            )

        return model, train_history
