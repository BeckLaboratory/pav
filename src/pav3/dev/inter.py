"""Interactive utilities."""

from pathlib import Path
from typing import Any

from .path import get_pav_dir

def init(
        path_pav: Path | str,
        path_init: Path | str,
        globals: dict[str, Any],
):
    """Initialize interactive session.

    :param path_pav: Path to PAV source directory. May be any file or directory in the PAV source
        including an symbolic link pointing to a resource in the PAV source.
    :param path_init: Path to initialization script.
    :param globals: Global namespace.

    :raises FileNotFoundError: If the initialization script does not exist.
    """
    with open(get_pav_dir(path_pav) / path_init, 'rt') as in_file:
        exec(in_file.read(), globals)
