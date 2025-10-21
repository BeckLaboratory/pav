"""General development utilities."""

__all__ = [
    'GlobalImport',
    'alias_to_obj_dict',
    'find_imports',
    'get_defined_names',
    'get_module_definitions',
]

import ast
from dataclasses import dataclass, field
import importlib
import inspect
import re
from types import ModuleType
from typing import Any, Generator, Optional


@dataclass(frozen=True)
class _GlobalImportScope:
    """Tracks one item in a stack of global imports.

    :ivar package_path: Name of the module where import statements are relative to (e.g. 'pav3.lgsv.interval' for
        imports in 'pav3/lgsv/interval.py').
    :ivar import_statements: A string of newline-separated import statements processed.
    :ivar global_dict: A dictionary mapping names of objects added to the global scope to the objects added. Used when
        removing a scope.
    """

    package_path: str
    import_statements: Optional[str] = field(repr=False)
    overwrite: bool
    caller_globals: dict[str, Any]
    mod_imports: bool = True
    mod_private: bool = True
    mod_dunder: bool = False
    global_dict: dict = field(default_factory=dict, init=False, repr=False)
    restore_dict: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Populate global_dict."""
        for alias, obj in alias_to_obj_dict(self.package_path, self.import_statements).items():
            self.global_dict[alias] = obj

    def apply(self) -> None:
        """Apply this scope to the global scope."""
        if not self.overwrite:
            for alias in set(self.global_dict.keys()) & (set(self.caller_globals.keys())):
                if self.caller_globals[alias] != self.global_dict[alias]:
                    raise ValueError(f'Alias "{alias}" is defined in globals')

        for alias in self.global_dict.keys():
            if alias in self.caller_globals:
                self.restore_dict[alias] = self.caller_globals[alias]
            self.caller_globals[alias] = self.global_dict[alias]

    def restore(self) -> None:
        """Restore the global scope to the state before this scope was applied."""
        for alias in self.global_dict.keys():
            if alias not in self.caller_globals or self.caller_globals[alias] != self.global_dict[alias]:
                continue

            if alias in self.restore_dict:
                self.caller_globals[alias] = self.restore_dict[alias]
            else:
                del self.caller_globals[alias]


class GlobalImport:
    """Mantains a stack of import frames."""

    def __init__(self) -> None:
        """Initialize impot stack."""
        self.import_stack = []
        self.caller_globals = inspect.currentframe().f_back.f_globals

    def push(
            self,
            package_path: str,
            import_statements: Optional[str] = None,
            overwrite: bool = True,
            mod_imports: bool = True,
            mod_private: bool = True,
            mod_dunder: bool = False,
    ) -> None:
        """Push a new scope to the global import stack and update the global scope.

        :param package_path: Name of the module where this import statement is found (e.g. 'pav3.lgsv.interval' if
            the import statements are in 'pav3/lgsv/interval.py').
        :param import_statements: A string of newline-separated import statements.
        :param overwrite: Whether to overwrite existing global variables.
        :param mod_imports: Whether to import objects from modules.
        :param mod_private: Whether to import private objects from modules.
        :param mod_dunder: Whether to import dunder objects from modules.
        """
        scope = _GlobalImportScope(
            package_path=package_path,
            import_statements=import_statements,
            overwrite=overwrite,
            caller_globals=self.caller_globals,
            mod_imports=mod_imports,
            mod_private=mod_private,
            mod_dunder=mod_dunder,
        )

        scope.apply()

        self.import_stack.append(scope)

    def pop(self) -> bool:
        """
        Pop the last scope from the stack and restore the global scope to the state before it was applied.

        :returns: True if a scope was popped, False if the global import stack is empty.
        """
        if not self.import_stack:
            return False

        self.import_stack.pop().restore()

        return True

    def refresh(self) -> None:
        """Undo and re-do all scopes.

        Run when objects might have changed.
        """
        old_state_stack = []

        while self.import_stack:
            old_state = self.import_stack.pop()
            old_state_stack.append(old_state)
            old_state.restore()

        while old_state_stack:
            old_state = old_state_stack.pop()
            self.push(old_state.package_path, old_state.import_statements, old_state.overwrite)

    def clear(self) -> None:
        """Clear all scopes."""
        while self.pop():
            pass

    def __len__(self) -> int:
        """Get the number of stack items."""
        return len(self.import_stack)

    def __bool__(self) -> bool:
        """Determine if the stack is non-empty."""
        return bool(self.import_stack)


def alias_to_obj_dict(
        package_path: str,
        import_statements: Optional[str] = None,
        mod_imports: bool = True,
        mod_private: bool = True,
        mod_dunder: bool = False,
) -> dict[str, Any]:
    """Find objects for relative imports.

    :param package_path: Name of the module where this import statement is found (i.e. 'pav3.lgsv.interval' if the).
    :param import_statements: A string of newline-separated import statements.
    :param mod_imports: Whether to import objects from modules.
    :param mod_private: Whether to import private objects from modules.
    :param mod_dunder: Whether to import dunder objects from modules.

    :returns: A set of new global names that were imported.
    """
    import_path = package_path.split('.')

    from_match_pattern = re.compile(r'^from\s+(\.*)([a-zA-Z_][a-zA-Z0-9_.]*)?\s+import\s+(.+)\s*$')
    as_pattern = re.compile(r'\s+as\s+')

    import_items = []  # List of tuples: (absolute import path, alias, original line)

    # Import statements
    if import_statements:
        for line in import_statements.split('\n'):
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            # Handle "from ... import ..." statements
            if from_match := from_match_pattern.match(line):
                dots, module_path, imports = from_match.groups()
                dot_count = len(dots)

                # Calculate absolute module path
                if dot_count == 0:
                    # Absolute import
                    abs_module = module_path

                else:
                    # Relative import
                    if dot_count > len(import_path):
                        raise ValueError(f"Attempted relative import beyond top-level package: {line}")

                    abs_module = (
                        (
                            '.'.join(
                                import_path[:-(dot_count)] if dot_count > 0 else import_path
                            )
                        ) + (
                            ('.' + module_path)
                                if module_path is not None else ''
                        )
                    )

                # Translate all imports
                import_set = {item.strip() for item in imports.split(',') if item.strip()}

                # Parse imports (handle multiple imports and aliases)
                #
                # * Module name
                # * Object name
                # * Alias
                # * Original line (string)
                import_items.extend([
                    (
                        abs_module,
                        items[0],
                        items[1] if len(items) > 1 else items[0],
                        line
                    )
                    for items in [
                        as_pattern.split(item.strip()) for item in import_set
                    ]
                ])

            else:
                raise ValueError(f'Unrecognized import structure: {line}')

    # Collect defined names
    if mod_imports:
        mod = importlib.import_module(package_path)

        for name, def_type, mod_id, mod_name in get_defined_names(mod):

            if name is None:
                continue  # Import statements, ignore (not from ... import)

            # Dunder
            if not mod_dunder and (name.startswith('__') and name.endswith('__')):
                continue

            # Private
            if not mod_private and name.startswith('_'):
                continue

            # Do import
            if def_type == 'import_from':
                if not mod_imports:
                    continue

                import_items.append((mod_id, mod_name, name, f'<from {mod_id} import {mod_name} as {name}>'))

            elif def_type in {'func', 'class', 'def'}:
                import_items.append((package_path, name, name, f'<{def_type} {name}>'))

    # Run imports
    global_dict = dict()

    for module_name, object_name, alias, line in import_items:

        try:
            obj = getattr(importlib.import_module(module_name, package_path), object_name)

        except ImportError as e:
            raise ImportError(f"Failed to import '{module_name}': line=\"{line}\"") from e

        except NameError as e:
            raise NameError(
                f"Failed to import '{module_name}.{object_name}': Object name not found in module: line=\"{line}\""
            ) from e

        if alias in global_dict and global_dict[alias] != obj:
            raise ValueError(
                f"Alias '{alias}' is used multiple times in stmts with conflicting objects: \"{line}\": "
                f"existing=\"{global_dict[alias]}\" (0x{id(global_dict[alias]):X}), new=\"{obj}\" (0x{id(obj):X})"
            )

        global_dict[alias] = obj

    # Return new global names
    return global_dict


def find_imports(
        module_name: str,
        public_only: bool = True,
        all_only: bool = False
) -> set[str]:
    """Locate objects defined in a module.

    :param module_name: Name of the module to inspect.
    :param public_only: Whether to include only public objects (i.e. not starting with '_').
    :param all_only: Whether to include only objects defined in '__all__'. Overrides public_only if __all__ has values.

    :returns: Set of object names defined in the module.
    """
    module = importlib.import_module(module_name)

    all_module_names = set(get_defined_names(module))

    defined_in_all = set(module.__all__) if hasattr(module, '__all__') else set()

    if missing_defs := defined_in_all - all_module_names:
        raise ValueError(
            f'Module "{module_name}" defines "__all__" with missing definitions: '
            f'{", ".join(sorted(missing_defs))}'
        )

    if hidden_defs := {name for name in defined_in_all if name.startswith('_')}:
        raise ValueError(
            f'Module "{module_name}" defines "__all__" with "hidden" definitions (starting with "_"): '
            f'{", ".join(sorted(hidden_defs))}'
        )

    if all_only:
        if not defined_in_all:
            raise ValueError(f'"all_only" was set, but module "{module_name}" does not define "__all__"')

        module_names = defined_in_all

    else:
        module_names = all_module_names

        if public_only:
            module_names = {name for name in module_names if not name.startswith('_')}

    return module_names


def get_defined_names(
        mod: ModuleType
):
    """
    Get names defined in a module.

    :param mod: Module to inspect.

    :yields: Tuples of (name, type, level) where "type" is "func", "class", "def", or "import". "level" is defined for
        imports (None for others) and is the level of the import with 0 being the top-level and 1 or more being a
        relative import.
    """

    tree = ast.parse(inspect.getsource(mod))

    for node in tree.body:

        if isinstance(node, ast.FunctionDef):
            yield node.name, 'func', None, None

        elif isinstance(node, ast.ClassDef):
            yield node.name, 'class', None, None

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                yield target.id, 'def', None, None

        elif isinstance(node, ast.AnnAssign):
            yield node.target.id, 'def', None, None

        elif isinstance(node, ast.ImportFrom):

            module_name = (
                '.'.join(mod.__name__.split('.')[:-node.level]
                + (node.module.split('.') if node.module else []))
            )

            for node_name in node.names:
                yield (
                    node_name.asname if node_name.asname else node_name.name,
                    'import_from',
                    module_name,
                    node_name.name,
                )

        elif isinstance(node, ast.Import):
            for node_name in node.names:
                yield None, 'import', node_name.name, node_name.name

        elif isinstance(node, ast.Expr):
            pass

        else:
            raise ValueError(f'Unknown type: {type(node)}')


# def get_defined_names(module: ModuleType) -> list[str]:
#     """Get a list of objects defined by the module ignoring dunder objects and imported modules."""
#     name_list = []
#
#     all_names = set(module.__all__) if hasattr(module, '__all__') else set()
#
#     for obj_name, obj in inspect.getmembers(module):
#
#         if obj_name in all_names:
#             name_list.append(obj_name)
#             continue
#
#         if obj_name.startswith('__'):
#             continue
#
#         if hasattr(obj, '__module__'):
#             if obj.__module__ == module.__name__:
#                 name_list.append(obj_name)
#         else:
#             if not isinstance(obj, ModuleType):
#                 name_list.append(obj_name)
#
#     return name_list


def get_module_definitions(
    module: ModuleType,
    include_dunder: bool = False
) -> Generator[str, None, None]:
    """Get names defined in a module object ignoring imports.

    :param module: Module object to inspect.
    :param include_dunder: Whether to include dunder objects (starting with '__').
    """
    source = inspect.getsource(module)
    tree = ast.parse(source)

    for node in tree.body:

        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if include_dunder or not node.name.startswith('__'):
                yield node.name

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if include_dunder or not target.id.startswith('__'):
                        yield target.id

