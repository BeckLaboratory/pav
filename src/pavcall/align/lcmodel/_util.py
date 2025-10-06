"""Utilities for loading low-confidence (LC) alignment models."""

__all__ = [
    'get_model',
    'locate_model',
    'locate_config_dir',
    'locate_config_dir_package',
    'locate_config_package',
    'null_model',
]

import json
import importlib
import importlib.resources.abc
import pathlib
from typing import Any, Optional

from ._lcmodel import LCAlignModel
from ._lcmodel_logistic import LCAlignModelLogistic
from ._lcmodel_null import LCAlignModelNull


def get_model(
        model_name: Optional[str] = None,
        search_dir: Optional[str] = None,
        search_dir_only: bool = True
) -> LCAlignModel | None:
    """Load model from path used to tag low-confidence (LC) alignments.

    If lc_model_path is None, an empty string, or keyword "none" (not case sensitive), a null model is used that does
    not tag any alignment records as LC.

    If lc_model_path is strictly alpha-numeric string, then the pre-trained models in the package data directory are
    searched first. If the model path contains non-alpha-numeric characters (e.g. '.' or './') or the model is not
    found in the package's pre-trained models, then this path is searched from the current working directory.

    If a model directory is located, "model.json" is read from it (an error is generated if it does not exist). This
    file contains at minimum the model type ("type") attribute. The model directory may contain other files read by the
    specific model type.

    If serach_dir is not None, it is searched first for the model. If it is not found, then

    :param model_name: Name of a pre-trained model or path to model directory. If None or 'none' (not case-sensitive),
        return a null model that does not tag any alignment records as low confidence.
    :param search_dir: Directory to search for pre-trained models. Can be used to locate custom models.
    :param search_dir_only: If True and search_dir is not None, only search in search_dir and fail if not found. If
            False, then continue searching in PAV pre-trained models. If search_dir is None, this parameter is ignored.

    :returns: LC alignment model or None if a model could not be located.

    :raises ValueError: If model path exists but does not contain a model definition file.
    :raises ValueError: If model definition is invalid or recursive aliases are detected.
    """
    if model_name is None:
        return null_model()

    if not isinstance(model_name, str):
        raise ValueError(f'LC align model path is not a string: {type(model_name)}')

    model_name = model_name.strip()

    if model_name.lower() == 'none' or model_name == '':
        return null_model()

    # Split model name into directory and model name (if any)
    model_name_path = pathlib.Path(model_name)

    if model_name_path.name != model_name:
        model_name = model_name_path.name
        search_dir = str(model_name_path.parent)
        search_dir_only = True

    # Locate model and load config
    model_tuple = locate_model(model_name, search_dir, search_dir_only)

    if model_tuple is None:
        return None

    model_dir, lc_model_def = model_tuple

    # Load model
    if lc_model_def['type'] == 'null':
        return null_model()
    elif lc_model_def['type'] == 'logistic':
        return LCAlignModelLogistic(lc_model_def, model_dir)
    else:
        raise ValueError(f'LC align model type is not supported: {lc_model_def["type"]}')


def locate_model(
        model_name: Optional[str] = None,
        search_dir: Optional[str] = None,
        search_dir_only: bool = True
) -> tuple[pathlib.Path | importlib.resources.abc.Traversable, dict[str, Any]] | None:
    """Locate a model by name.

    Recursively follow aliases until a model definition is found or no model can be located.

    :param model_name: Name of a pre-trained model or path to model directory. If None or 'none' (not case-sensitive),
        return a null model that does not tag any alignment records as low confidence.
    :param search_dir: Directory to search for pre-trained models. Can be used to locate custom models.
    :param search_dir_only: If True and search_dir is not None, only search in search_dir and fail if not found. If
        False, then continue searching in PAV pre-trained models. If search_dir is None, this parameter is ignored.

    :returns: A tuple containing two elements (or None if no model is found):
        1. The model directory (pathlib.Path or importlib.resources.abc.Traversable)
        2. The model definition (JSON)

    :raises ValueError: If model path exists but does not contain a model definition file.
    :raises ValueError: If model definition is invalid or recursive aliases are detected.
    """
    # Locate a model
    model_name_list = []

    while True:

        if model_name in model_name_list:
            raise ValueError(
                f'LC align model path contains recursive aliases: {model_name}: '
                f'aliases = {" -> ".join(model_name_list)}'
            )

        model_name_list.append(model_name)

        # Locate model directory and definition
        if (model_tuple := locate_config_dir(search_dir, model_name)) is not None:
            pass

        elif search_dir is None or not search_dir_only:

            if (model_tuple := locate_config_package(model_name)) is not None:
                pass

            else:
                model_tuple = locate_config_dir_package(model_name)

        if model_tuple is None:
            return None

        model_dir, model_config_filename = model_tuple

        # Load model definition
        with open(model_config_filename, 'r') as model_config_file:
            lc_model_def = json.load(model_config_file)

        # Check for alias
        alias = lc_model_def.get('alias', None)

        # Not an alias, get model
        if alias is None:
            if lc_model_def.get('type', None) is None:
                raise ValueError(
                    f'LC align model definition is missing "type" attribute for model with name "{model_name}": '
                    f'{model_config_filename}: aliases = {" -> ".join(model_name_list)}'
                )

            model_type = str(lc_model_def['type']).strip().lower()

            if model_type == '':
                raise ValueError(
                    f'LC align model definition has empty "type" attribute for model with name "{model_name}": '
                    f'{model_config_filename}: aliases = {" -> ".join(model_name_list)}'
                )

            return model_dir, lc_model_def

        # Process alias
        alias = str(alias).strip()

        if alias == '':
            raise ValueError(
                f'LC align model definition has empty "alias" attribute for model with name "{model_name}": '
                f'{model_config_filename}: aliases = {" -> ".join(model_name_list)}'
            )

        if (extra_keys := set(lc_model_def.keys()) - {'type', 'alias', 'description', 'name'}) != set():
            raise ValueError(
                f'LC align model definition has unexpected keys "{", ".join(sorted(extra_keys))}" for model with name '
                f'"{model_name}": {model_config_filename}: aliases = {" -> ".join(model_name_list)}'
            )

        model_name = alias


def locate_config_dir(
        module_dir: Optional[str | pathlib.Path],
        model_name: str
) -> tuple[pathlib.Path, pathlib.Path] | None:
    """Locate a model directory on disk.

    :param module_dir: Module directory
    :param model_name: Model name

    :returns: Tuple of paths (model directory, model config) or None if a model directory was not found.

    :raises ValueError: If the path to the model directory exists but is not a directory
    :raises ValueError: If the path to the model directory exists but the model config file is missing.
    """
    if module_dir is None:
        return None

    if not isinstance(module_dir, pathlib.Path):
        module_dir = pathlib.Path(module_dir).resolve()

    model_dir = module_dir / model_name

    if model_dir.exists():
        if not model_dir.is_dir():
            raise ValueError(f'LC align model directory exists but is not a directory: {model_dir}')

        model_config_filename = model_dir / 'model.json'

        if not model_config_filename.is_file():
            raise ValueError(
                f'LC align model directory exists model.json is missing or is not a regular file: '
                f'{model_config_filename}'
            )

        return model_dir, model_config_filename

    return None


def locate_config_package(
        model_name: str
) -> tuple[importlib.resources.abc.Traversable, importlib.resources.abc.Traversable] | None:
    """Locate a model directory in a package.

    :returns Tuple of paths (model directory, model config) or None if a model directory was not found.

    :raises ValueError: If the path to the model directory exists but is not a directory
    :raises ValueError: If the path to the model directory exists but the model config file is missing.
    """
    model_dir = importlib.resources.files('data.lcmodel').joinpath(model_name)

    if model_dir.is_dir() or model_dir.is_file():

        if not model_dir.is_dir():
            raise ValueError(f'LC align model directory exists but is not a directory: {model_dir}')

        model_config_filename = model_dir / 'model.json'

        if not model_config_filename.is_file():
            raise ValueError(
                f'LC align model directory exists model.json is missing or is not a regular file: '
                f'{model_config_filename}'
            )

        return model_dir, model_config_filename

    return None


def locate_config_dir_package(
        model_name: str
) -> tuple[pathlib.Path, pathlib.Path] | None:
    """Locate the configuration directory within a package.

    :param model_name: Model name.

    :returns: Tuple of paths (model directory, model config) or None if a model directory was not found.
    """
    last_path = pathlib.Path(__file__)
    module_dir = last_path.parent

    while module_dir != last_path and module_dir.name != 'src':
        last_path, module_dir = module_dir, module_dir.parent

    if module_dir.name == 'src':
        return locate_config_dir(model_name, module_dir / 'data' / 'lcmodel')

    return None


def null_model() -> LCAlignModelNull:
    """Get a null model."""
    return LCAlignModelNull(None, None)
