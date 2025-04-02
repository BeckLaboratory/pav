"""
Logic to run training stages.
"""

import collections

import argparse
import abc
import gzip
import json
import logging
import os
import uuid

logger = logging.getLogger(__name__)

class Stage(object, metaclass=abc.ABCMeta):

    STAGE_CACHE_FILENAME_BASE = 'stage_cache.json.gz'

    def __init__(self, stage_name: str, lctrain_config: dict, workdir_root: str, outdir: str):

        if lctrain_config is None:
            raise ValueError('lctrain_config is None')

        if 'name' not in lctrain_config:
            raise ValueError(f'Train config is missing a name for this model')

        self.model_name = lctrain_config['name']
        self.stage_name = stage_name
        self.lctrain_config = lctrain_config
        self.workdir_root = os.path.join(workdir_root, self.model_name)
        self.workdir = str(os.path.join(self.workdir_root, self.stage_name))

        self.stage_cache_filename = self.workdir_path(Stage.STAGE_CACHE_FILENAME_BASE)

        if outdir is None:
            outdir = '.'

        outdir = outdir.strip()

        if not outdir:
            outdir = '.'

        self.outdir = os.path.join(outdir, self.model_name)

    @abc.abstractmethod
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
        raise NotImplementedError

    def workdir_path(self,
                     filename: str,
                     stage_name: str=None,
                     subdir: str=''
                     ) -> str:
        """
        Get the path to a file in the working directory.

        :param filename: The name of the file.
        :param stage_name: The name of the stage file belongs to. If None, use this the name of this stage.
        :param subdir: Subdirectory within the stage directory.
        """

        file_root = self.workdir if stage_name is None else os.path.join(self.workdir_root, stage_name)

        return os.path.join(file_root, subdir, filename)

    def get_lctrain_config(self) -> dict:
        """
        Load the training configuration from the stage directory.
        """

        lctrain_config_filename = self.workdir_path('lctrain_config.json.gz', 'init')

        if not os.path.exists(lctrain_config_filename):
            raise RuntimeError(f'Missing staged configuration file: {lctrain_config_filename}')

        with gzip.open(lctrain_config_filename, 'rt') as in_file:
            return json.load(in_file)

    def get_stage_version_uuids(self, *args: str) -> dict[str, str]:
        """
        Get a dictionary of UUIDs belonging to each stage. This method reads the stage cache, and returns a dictionary
        of UUIDs belonging to that stage. UUIDs a stage stores for other stages are omitted (e.g. "init_train" is
        retrieved only from the "init" stage cache and is ignored in the "features" stage cache).

        :param args: List of stage names.
        """

        stage_uuids = dict()

        for stage in args:
            # Check input
            stage_cache_filename = self.workdir_path(Stage.STAGE_CACHE_FILENAME_BASE, stage)

            if not os.path.isfile(stage_cache_filename):
                raise RuntimeError(f'Missing stage cache file for UUIDs in stage {stage}: {stage_cache_filename}')

            with gzip.open(stage_cache_filename, 'rt') as f:
                stage_uuids.update({
                    key: (val if val != 'NA' else None)
                        for key, val in json.load(f).get('stage_uuid', {}).items()
                            if key.startswith(stage)
                })

        return stage_uuids

    def create_stage_uuid(self, **kwargs) -> dict[str, str]:
        """
        Create a dictionary of UUIDs for this stage. If the stage name is not included in kwargs, a UUID is generated
        for it.
        """

        if self.stage_name not in kwargs:
            kwargs[self.stage_name] = uuid.uuid4().hex

        return {
            key: (str(val) if val is not None else 'NA')
                for key, val in kwargs.items()
        }