"""
Stage: Eval
"""

import logging
import numpy as np
import pandas as pd

import pavlib
import svpoplib

import lctrain.util

from .stage import *
from .stage_init import StageInit

logger = logging.getLogger(__name__)

class StageEval(Stage):
    """Evaluate model performance."""

    STAGE_FILE_NAMES = {
        'stats_per_source_csv': 'stats_per_source_{name}.csv.gz',
        'stats_per_source_xlsx': 'stats_per_source_{name}.xlsx',
        'stats_per_asm_csv': 'stats_per_asm_{name}.csv.gz',
        'stats_per_asm_xlsx': 'stats_per_asm_{name}.xlsx'
    }

    def __init__(self, lctrain_config: dict, workdir_root: str, outdir: str):
        super().__init__('eval', lctrain_config, workdir_root, outdir)

        self.stats_per_source_csv = self.workdir_path(self.STAGE_FILE_NAMES['stats_per_source_csv'].format(name=lctrain_config.get('name', 'UNKNOWN')))
        self.stats_per_source_xlsx = self.workdir_path(self.STAGE_FILE_NAMES['stats_per_source_xlsx'].format(name=lctrain_config.get('name', 'UNKNOWN')))

        self.stats_per_asm_csv = self.workdir_path(self.STAGE_FILE_NAMES['stats_per_asm_csv'].format(name=lctrain_config.get('name', 'UNKNOWN')))
        self.stats_per_asm_xlsx = self.workdir_path(self.STAGE_FILE_NAMES['stats_per_asm_xlsx'].format(name=lctrain_config.get('name', 'UNKNOWN')))

        self.check_files = [
            self.stats_per_source_csv,
            self.stats_per_source_xlsx,
            self.stats_per_asm_csv,
            self.stats_per_asm_xlsx
        ]

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
        uuid_init_eval = None
        uuid_train = None

        data_config_cached = None

        eval_is_complete = None

        # Read stage cache
        if os.path.isfile(self.stage_cache_filename):
            with gzip.open(self.stage_cache_filename, 'rt') as f:
                stage_cache = json.load(f)

            uuid_init_eval = stage_cache.get('stage_uuid').get('init_eval', None)
            uuid_train = stage_cache.get('stage_uuid').get('train', None)

            data_config_cached = stage_cache.get('data', None)

        # Load data config
        data_config = self.lctrain_config['data']

        # Load init stage uuid
        stage_version_uuids = self.get_stage_version_uuids('init', 'train')

        # Determine if this stage is complete
        if uuid_train is None or uuid_train != stage_version_uuids.get('train', None):
            if uuid_train is not None:
                logger.info('Stage Eval: Detected mismatch with train stage: Re-running')

            uuid_train = stage_version_uuids.get('train')

            if uuid_train is None:
                raise RuntimeError('Missing UUID for train stage in train cache')

            eval_is_complete = False

        if uuid_init_eval is None or uuid_init_eval != stage_version_uuids.get('init_eval', None):
            if eval_is_complete is None:
                logger.info('Stage Eval: Detected mismatch with init stage (eval step): Re-running')

            uuid_init_eval = stage_version_uuids.get('init_eval')

            if uuid_init_eval is None:
                raise RuntimeError('Missing UUID for init stage (eval step) in init cache')

            eval_is_complete = False

        # Determine if this stage is complete
        if eval_is_complete is None:
            eval_is_complete = self.is_data_files_complete(data_config, data_config_cached)

        # Skip stage
        if not force and eval_is_complete:
            logger.debug('Stage Eval: All output is current: Skipping stage')
            return False
        elif force:
            logger.debug('Stage Eval: Forced execution')

        # Run eval
        logger.info('Stage Eval: Running...')

        if not pretend:

            # Load model
            try:
                logger.debug('Stage Eval: Loading LC model...')
                lc_model = pavlib.align.lcmodel.get_model(self.outdir)
            except Exception as e:
                raise RuntimeError(f'Stage eval: Failed to load LC model: {e}')

            logger.debug('Stage Eval: Reading data sources...')
            df_runpath = pd.read_csv(self.workdir_path(StageInit.STAGE_FILE_NAMES['eval_runpath_filename'], 'init'))
            df_datasource = pd.read_csv(self.workdir_path(StageInit.STAGE_FILE_NAMES['eval_datasource_filename'], 'init'))

            # Iterate over data sources
            df_per_source, df_per_asm = self.get_stat_tables(data_config, df_datasource, lc_model)

            # Write stats tables
            os.makedirs(self.workdir, exist_ok=True)

            df_per_source.to_csv(self.stats_per_source_csv, index=False, compression='gzip')
            df_per_source.to_excel(self.stats_per_source_xlsx, index=False)

            df_per_asm.to_csv(self.stats_per_asm_csv, index=False, compression='gzip')
            df_per_asm.to_excel(self.stats_per_asm_xlsx, index=False)

            # Write data configuration
            with gzip.open(self.stage_cache_filename, 'wt') as f:
                json.dump(
                    {
                        'stage_uuid': self.create_stage_uuid(
                            **{
                                'init_eval': uuid_init_eval,
                                'train': uuid_train,
                            }
                        ),
                        'data': data_config,
                    }, f, indent=4
                )

        return True

    def get_stat_tables(self, data_config, df_datasource, lc_model):
        """
        Get assembly stats tables.

        :param data_config: Data config.
        :param df_datasource: Data source dataframe.
        :param lc_model: LC model.
        """

        df_per_source_list = list()
        df_per_asm_all_list = list()

        for i in range(len(data_config)):
            data_source = data_config[i]
            data_source_name = data_source['name']

            logger.debug('Stage eval: %s (index %d)', data_source_name, i)

            df_data = df_datasource[df_datasource['name'] == data_source_name].reset_index(drop=True)

            if df_data.empty:
                raise RuntimeError(f'Data source {data_source_name} has no samples')

            df_per_asm_list = list()

            for data_index, row_datasource in df_data.iterrows():

                logger.debug('Running model for data source (name=%s, assembly=%s, hap=%s)', row_datasource['name'], row_datasource['asm_name'], row_datasource['hap'])

                # Read alignments
                df_align = pd.read_csv(
                    row_datasource['align_trim_none'], sep='\t',
                    dtype={'#CHROM': str}
                )

                df_align['QRY_LEN'] = df_align['QRY_END'] - df_align['QRY_POS']

                # Model predictions
                y_hat = lc_model(
                    df=df_align,
                    existing_score_model=lctrain.features.get_existing_score_model_name(row_datasource),
                    qry_fai=svpoplib.ref.get_df_fai(row_datasource['fai_path'])
                )

                # Get stats
                df_pass = df_align.loc[~ y_hat]
                df_fail = df_align.loc[y_hat]

                df_per_asm_list.append(
                    pd.Series(
                        [
                            row_datasource['name'], row_datasource['train'], row_datasource['asm_name'], row_datasource['hap'],
                            df_align.shape[0],
                            df_pass.shape[0], df_pass['QRY_LEN'].sum(),
                            df_pass['QRY_LEN'].mean(), df_pass['QRY_LEN'].median(), df_pass['QRY_LEN'].min(), df_pass['QRY_LEN'].max(),
                            df_fail.shape[0], df_fail['QRY_LEN'].sum(),
                            df_fail['QRY_LEN'].mean(), df_fail['QRY_LEN'].median(), df_fail['QRY_LEN'].min(), df_fail['QRY_LEN'].max()
                        ],
                        index=[
                            'name', 'train', 'asm_name', 'hap',
                            'n',
                            'pass_n', 'pass_bp',
                            'pass_bp_mean', 'pass_bp_med', 'pass_bp_min', 'pass_bp_max',
                            'fail_n', 'fail_bp',
                            'fail_bp_mean', 'fail_bp_med', 'fail_bp_min', 'fail_bp_max'
                        ]
                    )
                )

            # Merge records
            df_per_asm = pd.concat(df_per_asm_list, axis=1).T
            df_per_asm_all_list.append(df_per_asm)

            if np.all(df_per_asm['train']):
                is_train = True
            elif not np.any(df_per_asm['train']):
                is_train = False
            else:
                is_train = np.nan  # Should never occur

            df_per_source_list.append(
                pd.Series(
                    [
                        data_source_name, is_train, df_per_asm.shape[0],
                        df_per_asm['pass_n'].mean(), df_per_asm['pass_n'].min(), df_per_asm['pass_n'].max(),
                        df_per_asm['pass_bp'].mean(), df_per_asm['pass_bp'].median(), df_per_asm['pass_bp'].min(), df_per_asm['pass_bp'].max(),
                        df_per_asm['fail_n'].mean(), df_per_asm['fail_n'].min(), df_per_asm['fail_n'].max(),
                        df_per_asm['fail_bp'].mean(), df_per_asm['fail_bp'].median(), df_per_asm['fail_bp'].min(), df_per_asm['fail_bp'].max()
                    ],
                    index=[
                        'name', 'train', 'n_asm',
                        'pass_n_mean', 'pass_n_min', 'pass_n_max',
                        'pass_bp_mean', 'pass_bp_med', 'pass_bp_min', 'pass_bp_max',
                        'fail_n_mean', 'fail_n_min', 'fail_n_max',
                        'fail_bp_mean', 'fail_bp_med', 'fail_bp_min', 'fail_bp_max'
                    ]
                )
            )

        # Merge records
        df_per_source = pd.concat(df_per_source_list, axis=1).T
        df_per_asm = pd.concat(df_per_asm_all_list, axis=0)

        return df_per_source, df_per_asm


    def is_data_files_complete(self, data_config, data_config_cached) -> bool:
        """
        Determine if output is complete.

        :param data_config: Data configuration in the current run.

        :return: True if output is complete and False if this step must be re-run.
        """

        if data_config is None:
            raise RuntimeError(f'Data config is None')

        if data_config_cached is None:
            logger.info('Stage eval incomplete: No data config cached')
            return False

        if lctrain.util.data_config_strip(data_config) != lctrain.util.data_config_strip(data_config_cached):
            logger.info('Stage eval incomplete: Detected changes to data config')
            return False

        # Check for missing files
        missing_files = [filename for filename in self.check_files if not os.path.exists(filename)]

        if missing_files:
            # Log if partial (don't log on first run)
            if len(missing_files) != len(self.check_files):
                logger.info('Stage eval incomplete: Missing files: %s', ", ".join(missing_files))

            return False

        return True

    @staticmethod
    def is_data_config_consistent(data_config_a, data_config_b):
        return [
            {
                key: val for key, val in config_item.items() if key not in {'comment'}
            } for config_item in data_config_a
        ] == [
            {
                key: val for key, val in config_item.items() if key not in {'comment'}
            } for config_item in data_config_b
        ]
