"""Path utilities for development environments."""

from pathlib import Path
import tomllib

def get_pav_dir(
        path: Path | str,
) -> Path:
    """Locate the PAV source directory from a path within it.

    :param path: Path within a PAV source directory or a symbolic link to a resource within the
        PAV source directory. The path is followed whenther or not it exists or is a broken
        symbolic link.

    :return: Path to the PAV source directory.

    :raises FileNotFoundError: If the path does not point to a resource within a PAV source
        directory.
    """

    if path is None:
        raise ValueError('path is None')

    path = Path(path)

    for parent in [path] + list(path.resolve().parents):
        toml_path = parent / 'pyproject.toml'

        if toml_path.is_file():
            with open(toml_path, 'rb') as toml_file:
                if (
                        tomllib.load(toml_file)
                        .get('project', {})
                        .get('name', None)
                        == 'pav3'
                ):
                    return parent

    raise FileNotFoundError(f'path {path} does not point to a resource within a PAV source directory')
