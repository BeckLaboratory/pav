"""
Utilities for LC model training.
"""

import argparse
import json
import os
import pandas as pd
import gzip
import logging
import sys

import lctrain
import pavlib

logger = logging.getLogger(__name__)

def read_lctrain_config(config_file_list):
    """
    Read configuration files for training an LC model. If parameters or defined in multiple files, later files override
    previous ones.

    :param config_file_list: List of paths to JSON files.

    :return: Configuration dict from top-level concatenation of config files.
    """

    # Read
    lctrain_config = dict()

    for lctrain_file in config_file_list:
        with PlainOrGzReader(lctrain_file) as in_file:
            lctrain_dict = json.load(in_file)

            if not isinstance(lctrain_dict, dict):
                raise RuntimeError(f'Config file {lctrain_file} is not a dictionary at its top level: {type(lctrain_dict)}')

            lctrain_config.update(lctrain_dict)

    # Check
    missing_keys = [key for key in ['name', 'type', 'data'] if key not in lctrain_config.keys()]

    if missing_keys:
        raise RuntimeError(f'Config file(s) are missing required keys: {", ".join(missing_keys)}')

    if not isinstance(lctrain_config['data'], list):
        raise RuntimeError(f'Config file(s) type for key "data" must be a list: {type(lctrain_config["data"])}')

    # Return config
    return lctrain_config

def load_data_sources(data_config):
    """
    Locate data from all defined data sources.

    :param data_config: Section "data" in the training configuration.

    :return: A tuple of two DataFrames. The first (df_runpath) has one row for each datasource including a path to
        the PAV run directory. The second (df_datasource) has one row for each sample/hap pair discovered
        in all data sources with paths to alignment tables and other resources.
    """

    PAV_RESULTS_PATHS = {
        'align_trim_none': 'results/{asm_name}/align/trim-none/align_qry_{hap}.bed.gz',
        'align_trim_qry': 'results/{asm_name}/align/trim-qry/align_qry_{hap}.bed.gz',
        'align_stats': 'results/{asm_name}/align/trim-none/stats_qry_{hap}.tsv.gz'
    }

    PAV_RESULTS_PATHS_KEYS = sorted(PAV_RESULTS_PATHS.keys())

    df_list = list()
    df_runpath_list = list()
    dataset_name_set = set()
    i = 0

    for data_source in data_config:
        i += 1

        unknown_keys = [key for key in data_source.keys() if key not in {'name', 'path', 'include', 'exclude', 'include_hap', 'exclude_hap', 'comment', 'eval'}]

        if unknown_keys:
            raise RuntimeError(f'Config file(s) have unknown keys in data[{i - 1}]: {", ".join(unknown_keys)}')

        # Get parameters
        dataset_name = data_source.get('name', f'pavrun_{i - 1}')

        if dataset_name in dataset_name_set:
            raise RuntimeError(f'Config file(s) have duplicate names in "data": {dataset_name}')

        dataset_name_set.add(dataset_name)

        is_eval = pavlib.util.as_bool(data_source.get('eval', False))

        pav_run_path = data_source.get('path', None)

        if pav_run_path is None:
            raise RuntimeError(f'Config file(s) for data source {dataset_name}: Missing required key: data[{i - 1}]/path')

        if not os.path.isdir(pav_run_path):
            raise RuntimeError(f'Config file(s) for data source {dataset_name}: Invalid path in data[{i - 1}]: {pav_run_path}')

        df_runpath_list.append(
            pd.Series(
                [dataset_name, not is_eval, pav_run_path, i - 1],
                index=['name', 'train', 'pav_run_path', 'data_source_index']
            )
        )

        #
        # Include/exclude samples
        #

        if 'include' in data_source and not isinstance(data_source['include'], list):
            raise RuntimeError(f'Config file(s) for data source {dataset_name}: Type for key data[{i - 1}]/include must be a list: {type(data_source["include"])}')

        include_assemblies = data_source.get('include', None)

        if include_assemblies is not None:
            include_assemblies = set(include_assemblies)

        if 'exclude' in data_source and not isinstance(data_source['exclude'], list):
            raise RuntimeError(f'Config file(s) for data source {dataset_name}: Config option data[{i - 1}]/exclude must be a list: {type(data_source["exclude"])}')

        exclude_assemblies = set(data_source.get('exclude', []))


        #
        # Include/exclude haps
        #

        if 'include_hap' in data_source and not isinstance(data_source['include_hap'], list):
            raise RuntimeError(f'Config file(s) for data source {dataset_name}: Type for key data[{i - 1}]/include_hap must be a list: {type(data_source["include_hap"])}')

        include_hap = data_source.get('include_hap', None)

        if include_hap is not None:
            include_hap = set(include_hap)

        if 'exclude_hap' in data_source and not isinstance(data_source['exclude_hap'], list):
            raise RuntimeError(f'Config file(s) for data source {dataset_name}: Config option data[{i - 1}]/exclude_hap must be a list: {type(data_source["exclude_hap"])}')

        exclude_hap = set(data_source.get('exclude_hap', []))


        # Load assembly table
        logger.debug('Loading data source: %s', dataset_name)

        config_filename = os.path.join(pav_run_path, 'config.json')

        if os.path.isfile(config_filename):
            config = json.load(open(config_filename))
        else:
            config = {}

        asm_table_filename = config.get('assembly_table', None)

        if asm_table_filename is None and os.path.isfile(os.path.join(pav_run_path, 'assemblies.tsv')):
            asm_table_filename = 'assemblies.tsv'

        if asm_table_filename is None and os.path.isfile(os.path.join(pav_run_path, 'assemblies.xlsx')):
            asm_table_filename = 'assemblies.xlsx'

        if asm_table_filename is None:
            raise RuntimeError(f'Config file(s) for data source {dataset_name}: No input assembly table in config ("assembly_table") and the default table filename was not found ("assemblies.tsv")')

        asm_table = pavlib.pipeline.read_assembly_table(asm_table_filename, config)

        asm_table = asm_table[[col for col in asm_table.columns if col.startswith('HAP_')]]

        asm_table.columns = [col[4:] for col in asm_table.columns]

        # Read samples
        for asm_name in asm_table.index:
            if (include_assemblies is not None and asm_name not in include_assemblies) or asm_name in exclude_assemblies:
                continue

            for hap in asm_table.columns:

                if (include_hap is not None and hap not in include_hap) or hap in exclude_hap:
                    continue

                # Locate FASTA and FAI
                fa_path = os.path.join(pav_run_path, asm_table.loc[asm_name, hap])

                if not os.path.isfile(fa_path):
                    raise RuntimeError(f'Config file(s) for data source {dataset_name}: Invalid PAV assembly path in assembly table (assembly="{asm_name}", hap="{hap}") in data[{i - 1}]: {fa_path}')

                fai_path = os.path.join(pav_run_path, asm_table.loc[asm_name, hap] + '.fai')

                if not os.path.isfile(fai_path):
                    raise RuntimeError(f'Config file(s) for data source {dataset_name}: Invalid PAV assembly path in assembly table has no FAI index (assembly="{asm_name}", hap="{hap}") in data[{i - 1}]: {fai_path}')

                # Locate PAV results
                pav_results_paths = {
                    key: os.path.join(pav_run_path, PAV_RESULTS_PATHS[key].format(asm_name=asm_name, hap=hap)) for key in PAV_RESULTS_PATHS
                }

                for val in pav_results_paths.values():
                    if not os.path.isfile(val):
                        raise RuntimeError(f'Config file(s) for data source {dataset_name}: Invalid PAV results path in assembly table (sample="{asm_name}", hap="{hap}") in data[{i - 1}]: {val}')

                # Save
                df_list.append(
                    pd.Series(
                        [
                            dataset_name, not is_eval, asm_name, hap,
                            fa_path, fai_path,
                        ] + [pav_results_paths[key] for key in PAV_RESULTS_PATHS_KEYS],
                        index=[
                            'name', 'train', 'asm_name', 'hap',
                            'fa_path', 'fai_path'
                        ] + PAV_RESULTS_PATHS_KEYS
                    )
                )

    # Concatenate dataframes
    df_runpath = pd.concat(df_runpath_list, axis=1).T
    df_datasource = pd.concat(df_list, axis=1).T

    missing_set = set(df_runpath['name']) - set(df_datasource['name'])

    if missing_set:
        logger.warning('Config file(s) for data set(s): Missing data source(s) in "data": %s', ', '.join(sorted(missing_set)))

    return df_runpath, df_datasource

def data_config_strip(data_config: list[dict], drop_eval: bool=False) -> list[dict]:
    """
    Strip comments and drop eval records from data config. Used to check data configurations for substantive changes
    across stages.

    :param data_config: A list of training data sources.
    :param drop_eval: Drop eval records (return train-only data sources).

    :return: A list of training data sources without comments.
    """
    return [
        {
            key: val for key, val in data_config_item.items() if key not in {'comment'}
        } for data_config_item in data_config if (
            not (drop_eval and pavlib.util.as_bool(data_config_item.get('eval', False)))
        )
    ]

class PlainOrGzReader:
    """
    Read a plain or a gzipped file using context guard.

    Code copied from SV-Pop (svpoplib.seq.PlainOrGzReader).

    Example:
        with PlainOrGzReader('path/to/file.gz'): ...
    """

    def __init__(self, file_name, mode='rt'):

        self.file_name = file_name

        self.is_gz = file_name.lower().endswith('.gz')
        self.mode = mode

        self.file_handle = None

    def __enter__(self):

        if self.is_gz:
            self.file_handle = gzip.open(self.file_name, self.mode)
        else:
            self.file_handle = open(self.file_name, self.mode)

        return self.file_handle

    def __exit__(self, exc_type, exc_value, traceback):

        if self.file_handle is not None:
            self.file_handle.__exit__()
            self.file_handle = None

#
# GPU functions
#

def tf_num_gpus():
    """
    Get the number of available GPUs.

    Code from: https://d2l.ai/chapter_builders-guide/use-gpu.html
    """

    import tensorflow as tf

    return len(tf.config.experimental.list_physical_devices('GPU'))

def tf_try_gpu(i=0):  #@save
    """
    Return gpu(i) if exists, otherwise return cpu().

    Code from: https://d2l.ai/chapter_builders-guide/use-gpu.html
    """

    if tf_num_gpus() > i:
        return tf_gpu(i)

    return tf_cpu()

def tf_cpu():  #@save
    """
    Get the CPU device.

    Code from: https://d2l.ai/chapter_builders-guide/use-gpu.html
    """

    import tensorflow as tf

    return tf.device('/CPU:0')

def tf_gpu(i=0):  #@save
    """
    Get a GPU device.

    Code from: https://d2l.ai/chapter_builders-guide/use-gpu.html
    """

    import tensorflow as tf

    return tf.device(f'/GPU:{i}')