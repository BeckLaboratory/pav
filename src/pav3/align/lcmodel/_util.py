"""Utilities for loading low-confidence (LC) alignment models."""

__all__ = [
    'get_model',
    'locate_model',
    'locate_config_package',
    'locate_config_filesystem',
    'null_model',
]

import json
import importlib
import importlib.resources.abc
import pathlib
import re
from typing import Any, Optional

from ...io import ResourceReader

from ._lcmodel import LCAlignModel
from ._lcmodel_logistic import LCAlignModelLogistic
from ._lcmodel_null import LCAlignModelNull

LC_MODEL_RESOURCE = 'pav3.data.lcmodel'
"""Resource directory where built-in LC models are kept in the package."""

def get_model(
        model_name: Optional[str] = None,
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

    # Locate model and load config
    model_tuple = locate_model(
        model_name=model_name,
    )

    if model_tuple is None:
        return None

    resource_type, anchor, lc_model_def = model_tuple

    # Load model
    if lc_model_def['type'] == 'null':
        return null_model()
    elif lc_model_def['type'] == 'logistic':
        return LCAlignModelLogistic(lc_model_def, resource_type, anchor)
    else:
        raise ValueError(f'LC align model type is not supported: {lc_model_def["type"]}')


def locate_model(
        model_name: Optional[str] = None,
) -> tuple[str, str, dict[str, Any]] | None:
    """Locate a model by name or path.

    Recursively follow aliases until a model definition is found or no model can be located.

    :param model_name: Name of a pre-trained model or path to model directory. If None or 'none' (not case-sensitive),
        return a null model that does not tag any alignment records as low confidence.

    :returns: A tuple containing two elements (or None if no model is found):

        1. The model directory (pathlib.Path or importlib.resources.abc.Traversable)
        2. The model definition (JSON)

    :raises ValueError: If model path exists but does not contain a model definition file.
    :raises ValueError: If model definition is invalid or recursive aliases are detected.
    """
    # Locate a model
    model_name_list = []

    iteration = 0
    max_iterations = 5

    while iteration < max_iterations:
        iteration += 1

        if model_name in model_name_list:
            raise ValueError(
                f'LC align model path contains recursive aliases: {model_name}: '
                f'aliases = {" -> ".join(model_name_list)}'
            )

        model_name_list.append(model_name)

        # Locate model
        model_tuple = None

        if not model_tuple and re.search(r'^\w+$', model_name):
            model_tuple = locate_config_package(model_name)

        if not model_tuple:
            model_tuple = locate_config_filesystem(model_name)

        if not model_tuple:
            return None

        # Read configuration
        resource_type, anchor = model_tuple

        # Load model definition
        with ResourceReader(anchor, 'model.json', resource_type) as in_file:
            lc_model_def = json.load(in_file)

        # Check for alias
        alias = lc_model_def.get('alias', None)

        # Not an alias, get model
        if alias is None:
            if lc_model_def.get('type', None) is None:
                raise ValueError(
                    f'LC align model definition is missing "type" attribute for model with name "{model_name}": '
                    f'{resource_type}:{anchor}:model.json aliases = {" -> ".join(model_name_list)}'
                )

            model_type = str(lc_model_def['type']).strip().lower()

            if model_type == '':
                raise ValueError(
                    f'LC align model definition has empty "type" attribute for model with name "{model_name}": '
                    f'{resource_type}:{anchor}:model.json: aliases = {" -> ".join(model_name_list)}'
                )

            return resource_type, anchor, lc_model_def

        # Process alias
        alias = str(alias).strip()

        if alias == '':
            raise ValueError(
                f'LC align model definition has empty "alias" attribute for model with name "{model_name}": '
                f'{resource_type}:{anchor}:model.json: aliases = {" -> ".join(model_name_list)}'
            )

        if (extra_keys := set(lc_model_def.keys()) - {'type', 'alias', 'description', 'name'}) != set():
            raise ValueError(
                f'LC align model definition has unexpected keys "{", ".join(sorted(extra_keys))}" for model with name '
                f'{resource_type}:{anchor}:model.json: aliases = {" -> ".join(model_name_list)}'
            )

        model_name = alias

    raise ValueError(f'Failed to locate LC model after {iteration} iterations, tried: {", ".join(model_name_list)}')


def locate_config_package(
        model_name: str
) -> Optional[tuple[str, str]]:
    """Locate a model directory in the PAV package.

    :returns: If a model is found, returns a tuple of (resource_type, anchor) where resource_type is "package"
        and anchor is the model directory's resource name. Returns None if a model directory was not found.
    """

    if not re.search(r'^\w+$', model_name):
        return None

    model_resource = LC_MODEL_RESOURCE + '.' + model_name

    model_dir = importlib.resources.files(model_resource)

    if not (model_dir.is_dir() or model_dir.is_file()):
        return None

    if not model_dir.is_dir():
        raise ValueError(f'LC align model directory resource exists but is not a directory: {model_dir}')

    if not importlib.resources.is_resource(model_resource, 'model.json'):
        raise ValueError(f'Missing "model.json" in LC align model resource: {model_resource}')

    return 'package', model_resource


def locate_config_filesystem(
        model_name: str
) -> Optional[tuple[str, str]]:
    """Locate an LC model on the filesystem.

    :param model_name: Model name.

    :returns: If a model is found, returns a tuple of (resource_type, anchor) where resource_type is "filesystem"
        and anchor is the path to the model directory. Returns None if a model directory was not found.
    """

    model_path = pathlib.Path(model_name)

    if not model_path.exists():
        return None

    if not model_path.is_dir():
        raise ValueError(f'LC align model directory exists but is not a directory: {str(model_path)}')

    if not (model_path / 'model.json').is_file():
        raise ValueError(f'Missing "model.json" in LC align model directory: {str(model_path)}')

    return 'filesystem', str(model_path)


def null_model() -> LCAlignModelNull:
    """Get a null model."""
    return LCAlignModelNull(None, None)
