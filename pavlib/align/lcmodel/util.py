"""
Utilities for loading low-confidence (LC) alignment models.
"""

import json
import os
import re

from .lcmodel_null import LCAlignModelNull

def get_model(lc_model_path=None, search_dir=None, alias_filename_list=None):
    """
    Load model from path used to tag low-confidence (LC) alignments.

    If lc_model_path is None, an empty string, or keyword "none" (not case sensitive), a null model is used that does
    not tag any alignment records as LC.

    If lc_model_path is strictly alpha-numeric string, then the pre-trained models in the PAV directory are searched
    first ("files/lcmodel/models" in the PAV directory). If the model path contains non-alpha-numeric characters
    (e.g. '.' or './') or the model is not found in PAV's pre-trained models, then this path is searched from the
    current working directory (i.e. PAV run directory).

    If a model directory is located, "model.json" is read from it (an error is generated if it does not exist). This
    file contains at minimum the model type ("type") attribute. The model directory may contain other files read by the
    specific model type.

    For more information about models and training custom models, see PAV directory "files/lcmodel/LC_MODEL.md".

    :param lc_model_path: Name of a pre-trained model or path to model directory. If None or 'none' (not case-
        sensitive), return a null model that does not tag any alignment records as low confidence.
    :param search_dir: Directory to search for pre-trained models. This is set to PAV's model directory by PAV, but
        could be used to resolve model paths by keywords for non-PAV models when used as an API outside of PAV.

    :return: LC alignment model.
    """

    import pavlib.align.lcmodel

    if alias_filename_list is None:
        alias_filename_list = list()

    # Check path and null-model cases
    if lc_model_path is None:
        return LCAlignModelNull()

    if not isinstance(lc_model_path, str):
        raise RuntimeError(f'LC align model path is not a string: {type(lc_model_path)}')

    lc_model_path = lc_model_path.strip()

    if lc_model_path.lower() == 'none' or lc_model_path == '':
        return LCAlignModelNull()

    if not isinstance(lc_model_path, str):
        raise RuntimeError(f'LC align model path is not a string: {type(lc_model_path)}')

    # Locate model directory from PAV
    found_model_path = None

    if re.search(r'^\w+$', lc_model_path):

        if search_dir is None:
            search_dir = '.'

        if not isinstance(search_dir, str):
            raise RuntimeError(f'LC align model search directory is not a string: {type(search_dir)}')

        search_dir = search_dir.strip()

        model_path_test = os.path.join(search_dir, lc_model_path)

        if os.path.exists(model_path_test):
            found_model_path = model_path_test

    if found_model_path is None:
        found_model_path = lc_model_path

    if not os.path.exists(found_model_path):
        raise RuntimeError(f'Cannot locate LC align model directory: {found_model_path}')

    if not os.path.isdir(found_model_path):
        raise RuntimeError(f'LC align model path is not a directory: {found_model_path}')

    model_def_filename = os.path.join(found_model_path, 'model.json')

    if not os.path.exists(model_def_filename):
        raise RuntimeError(f'Cannot locate LC align model definition JSON file: {model_def_filename}')

    if not os.path.exists(model_def_filename):
        raise RuntimeError(f'LC align model definition JSON is not a regular file: {model_def_filename}')

    # Check for recursive alias
    if model_def_filename in alias_filename_list:
        alias_path_str = ' -> '.join([f'"{f}"' for f in alias_filename_list])
        raise RuntimeError(f'Recursive path in LC align model aliases: {model_def_filename}: Alias path {alias_path_str}')

    # Read model definition JSON
    try:
        with open(model_def_filename, 'r') as f:
            lc_model_def = json.load(f)
    except Exception as e:
        raise RuntimeError(f'Cannot read LC align model definition JSON file: {model_def_filename}: {e}')

    # Check for alias
    if 'alias' in lc_model_def:

        unknown_set = set(lc_model_def.keys()) - {'alias', 'name'}

        if unknown_set:
            n_unknown = len(unknown_set)
            unknown_str = ', '.join(sorted(unknown_set)[:3]) + ('...' if n_unknown > 3 else '')
            raise RuntimeError(f'LC align model definition JSON specifies an alias, which may only have the "alias" attribute: Found {n_unknown} unknown attributes "{unknown_str}": {model_def_filename}')

        alias_filename_list.append(model_def_filename)

        return get_model(lc_model_def['alias'], search_dir=search_dir, alias_filename_list=alias_filename_list)

    # Check and set model path
    if 'model_path' in lc_model_def:
        raise RuntimeError(f'LC align model definition JSON file contains the reserved "model_path" attribute: {model_def_filename}')

    lc_model_def['model_path'] = found_model_path

    # Get type
    model_type = lc_model_def.get('type', None)
    if model_type is None:
        raise RuntimeError(f'LC align model definition JSON file is missing "type" attribute: {model_def_filename}')

    model_type = str(model_type).strip()

    # Get name
    model_name = lc_model_def.get('name', None)

    if model_name is None or model_name.strip() == '':
        model_name = '<MODEL_NAME_NOT_SPECIFIED>'
    else:
        model_name = str(model_name).strip()

    # Locate model class
    model_name_camel = 'LCAlignModel' + (model_type[0].upper() + model_type[1:].lower())

    try:
        model_class = getattr(pavlib.align.lcmodel, model_name_camel)
    except AttributeError:
        raise RuntimeError(f'Cannot locate LC align model class for definition {model_name} with model type "{model_type}": {model_name_camel}')

    lc_model = model_class(lc_model_def)

    lc_model.check_known_attributes()

    return lc_model