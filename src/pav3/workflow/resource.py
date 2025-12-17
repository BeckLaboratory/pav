"""Objects for passing resources across tasks.

Objects of type `ResourceContainer` are passed across workflow tasks and contain information about
the workflow and resources needed for each task. A task will consume a `ResourceContainer`, utilize
the resources within it to execute the task, then produce a `ResourceContainer` as output. While
tasks can have arguments of their own, resource containers allow resources to be coupled with tags
and provide ways of selecting the appropriate resources for each step of the task. They also
provide a way of providing resources where the number or types of arguments might vary.

For example, a task merging multiple callset tables might extract all resources tagged with
"type = callset" and merge based on attribute tag "vartype". Further, it might use tags associated
with each resource to provide additional arguments to the task, such as a list of identifiers
constructed for each.

For example, a resource might look like:

    resources = ResourceContainer(
        'gen_task',
        paths=(
            WorkflowResource(
                'call_h1',
                Path('results/HG00171/call_hap/call_insdel_h1.parquet'),
                {'type': 'callset', 'hap': 'h1'}
            ),
            WorkflowResource(
                'call_h2',
                Path('results/HG00171/call_hap/call_insdel_h2.parquet'),
                {'type': 'callset', 'hap': 'h2'}
            ),
        ),
        tags={
            'asm_name': 'HG00171',
            'vartype': 'insdel'
        }
    )

    [
        (
            pl.scan_parquet(resource.value),
            f'{c.tags["asm_name"]}_{resource.tags["hap"]}',
        ) for resource in c.paths.resource_by_tag({'type': 'callset'})
    ]

Then, the input from merging can be constructed easily:

    (
        (
            pl.scan_parquet(resource.value),
            f'{c.tags["asm_name"]}_{resource.tags["hap"]}',
        ) for resource in c.paths.resource_by_tag({'type': 'callset'})
    )

This paradigm makes is possible to set resources and attributes at the beginning of the pipeline
and pass them through subsequent tasks. For example, the output task would construct a new
resource object pointing to the merged callset table, but it will set "asm_name" by copying from
its input resource.
"""

__all__ = [
    'Taggable',
    'WorkflowResource',
    'MappableResource',
    'ResourceContainer',
]

from collections import ChainMap
from collections.abc import (
    Iterable,
    Mapping,
)
from pathlib import Path
from typing import (
    Any,
    Optional,
    TypeGuard,
    TypeVar,
    overload,
)

from agglovar.meta.decorators import immutable
from agglovar.meta.descriptors import (
    CheckedString,
)

T = TypeVar('T')

class Taggable:
    """Object that can be matched against tags.

    Each tag is a key-value pair where keys and values are strings. For example, key might be
    "vartype" and value might be "insdel" or "snv" to match resources for different variant types.

    :ivar tags: Tags for the object.
    """
    tags: Mapping[str, str]

    def __init__(
            self,
            tags: Optional[Mapping[str, str]] = None,
    ):
        """Initialize the taggable object.

        :param tags: Tags for the object.
        """
        self.tags = dict(tags) if tags is not None else {}

    def match_tag(
            self,
            key: str,
            value: str | Iterable[str],
    ):
        """Determine if this resource matches a tag.

        :param key: Name of the tag.
        :param value: Value of the tag. Value may be a single tag value (str) or an iterable of tag
            values (True if any match).
        """
        if key is None or value is None or key not in self.tags:
            return False

        if isinstance(value, str):
            return self.tags.get(key) == value
        else:
            return self.tags.get(key) in set(value)

    def check_tags(
            self,
            tags: str | Iterable[str]
    ) -> None:
        """Check if keys exist in the resource map and raise an error if not."""

        if isinstance(tags, str):
            tags = {tags,}

        if missing_key := set(tags) - set(self.tags.keys()):
            n = len(missing_key)
            missing_key = ', '.join(sorted(missing_key)[:3]) + ('...' if n > 3 else '')
            resource_name = getattr(self, 'task_name', '<unknown>')

            raise KeyError(
                f'Resource {resource_name} ({type(self).__name__}): Missing {n} tag(s): {missing_key}'
            )

    def __repr__(self):
        return f'Taggable({', '.join(self.tags.keys())})'


@immutable
class WorkflowResource[T](Taggable):
    """Resource object for use in workflows.

    :ivar key: Name of the resource.
    :ivar value: Value of this resource.
    """
    key: str = CheckedString(1, match=r'^[a-zA-Z0-9_]+$')
    value: T

    def __init__(
            self,
            key: str,
            value: T,
            tags: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Initialize the resource object.

        :param key: Name of the resource.
        :param value: Resource object.
        :param tags: Tags for the resource.
        """
        super().__init__(tags)

        self.key = key
        self.value = value

    def __repr__(self):
        return f'WorkflowResource({self.key}, "{type(self.value)}", {super(Taggable).__repr__()})'


class MappableResource[T](Mapping[str, T]):
    """An interface for mappable resources."""
    _resource_map: Mapping[str, T]

    def __init__(
            self,
            resource_map: Mapping[str, T],
    ):
        """Initialize the resource map.

        :param resource_map: Resources this map will access.
        """
        self._resource_map = resource_map if resource_map is not None else {}

    @overload
    def __getitem__(self, key: str | int) -> T:...

    @overload
    def __getitem__(self, key: slice | Iterable[str | int]) -> list[T]:...

    def __getitem__(
            self,
            key: str | int | slice | Iterable[str | int]
    ) -> T | list[T]:
        """Get a resource value by name or index.

        :param key: Key to get resource by. May be a string, integer, slice, or iterable of strings or integers.
            A string or integer returns a single resource, other types return a list of resources.

        :return: Resource value or list of resource values.
        """

        if isinstance(key, (int, str)):
            return self._get_item(key).value

        elif isinstance(key, slice):
            return [
                self._get_item(k).value for k in iter(key.indices(len(self)))
            ]

        else:
            return [
                self._get_item(k).value for k in key
            ]

    @overload
    def get_resource(self, key: str | int) -> WorkflowResource[T]:...

    @overload
    def get_resource(self, key: slice | Iterable[str | int]) -> list[WorkflowResource[T]]:...

    def get_resource(
            self,
            key: str | int | slice | Iterable[str | int]
    ) -> WorkflowResource[T] | list[WorkflowResource[T]]:
        """Get a resource by name or index.

        :param key: Key to get resource by. May be a string, integer, slice, or iterable of strings or integers.
            A string or integer returns a single resource, other types return a list of resources.

        :return: Resource or list of resources.
        """
        if isinstance(key, (int, str)):
            return self._get_item(key)

        elif isinstance(key, slice):
            return [
                self._get_item(k) for k in iter(key.indices(len(self)))
            ]

        else:
            return [
                self._get_item(k) for k in key
            ]

    @overload
    def by_tag(self, key: dict[str, str]) -> list[T]:...

    @overload
    def by_tag(self, key: str, value: str | Iterable[str]) -> list[T]:...

    @overload
    def by_tag(self, key: str, value: str | Iterable[str], intersect: bool = True) -> list[T]:...

    def by_tag(
            self,
            key: str | Mapping[str, str | Iterable[str]],
            value: Optional[str | Iterable[str]] = None,
            intersect: bool = True
    ) -> list[T]:
        """Get resource values matching a tag.

        :param key: Name of the tag.
        :param value: Value of the tag. Value may be a single tag value (str) or an iterable of tag
            values (True if any match).
        :param intersect: If True and key is a dictionary of keys and values, a resource must match
            all keys and values in the dict (intersect operation). If False, a resource must match
            at least one key and value in the dict (union operation).

        :return: List of resources that match the tag. If no resources match, returns an empty list.
        """

        return [
            resource.value for resource in self.resource_by_tag(key, value, intersect)
        ]

    @overload
    def resource_by_tag(
            self,
            key: dict[str, str],
    ) -> list[WorkflowResource[T]]:...

    @overload
    def resource_by_tag(
            self,
            key: str,
            value: str | Iterable[str]
    ) -> list[WorkflowResource[T]]:...

    @overload
    def resource_by_tag(
            self,
            key: str,
            value: str | Iterable[str],
            intersect: bool = True,
    ) -> list[WorkflowResource[T]]:...

    def resource_by_tag(
            self,
            key: str | Mapping[str, str | Iterable[str]],
            value: Optional[str | Iterable[str]] = None,
            intersect: bool = True
    ) -> list[WorkflowResource[T]]:
        """Get resources matching a tag.

        :param key: Name of the tag.
        :param value: Value of the tag. Value may be a single tag value (str) or an iterable of tag
            values (True if any match).
        :param intersect: If True and key is a dictionary of keys and values, a resource must match
            all keys and values in the dict (intersect operation). If False, a resource must match
            at least one key and value in the dict (union operation).

        :return: List of resources that match the tag. If no resources match, returns an empty list.
        """

        if isinstance(key, str):
            if not isinstance(value, str):
                value = set(value)  # Avoid exhausting an iterator on the first round.

            keys = self._keys_with_tag(key, value)

        else:
            key_list = []

            if value is not None:
                raise ValueError('value must be None when key is a dictionary')

            for k, v in key.items():
                key_list.append(self._keys_with_tag(k, v))

            if not key_list:
                return []

            if intersect:
                keys = set.intersection(*key_list)
            else:
                keys = set.union(*key_list)

        return [
            resource for resource in self._resource_map.values()
            if resource.key in keys
        ]

    def _keys_with_tag(
            self,
            key: str,
            value: str | Iterable[str],
    ) -> set[str]:
        if not isinstance(value, str):
            value = set(value)  # Avoid exhausting an iterator on the first round.

        return {
            resource.key for resource in self._resource_map.values()
            if resource.match_tag(key, value)
        }

    def __iter__(self):
        return (resource.value for resource in self._resource_map.values())

    def __len__(self):
        return len(self._resource_map)

    def __contains__(self, key: str):
        return key in self._resource_map

    def keys(self) -> Iterable[str]:
        return self._resource_map.keys()

    def values(self) -> Iterable[T]:
        return self._resource_map.values()

    def items(self) -> Iterable[tuple[str, T]]:
        return self._resource_map.items()

    def resource(
            self,
            key: str,
    ) -> WorkflowResource[T]:
        """Retrieve a resource object by key."""
        return self._resource_map[key]

    def check_keys(
            self,
            key: str | Iterable[str]
    ) -> None:
        """Check if keys exist in the resource map and raise an error if not."""

        if isinstance(key, str):
            key = {key,}

        if missing_key := set(key) - set(self._resource_map.keys()):
            n = len(missing_key)
            missing_key = ', '.join(sorted(missing_key)[:3]) + ('...' if n > 3 else '')
            resource_name = getattr(self, 'task_name', '<unknown>')

            raise KeyError(
                f'Resource {resource_name} ({type(self).__name__}): Missing {n} key(s): {missing_key}'
            )

    def _get_item(
            self,
            key: str | int
    ) -> WorkflowResource[T]:
        """Get a resource by name or index.

        :param key: Key to get resource by. May be a string or integer.

        :return: Resource.
        """
        if isinstance(key, int):
            if key < 0 or key >= len(self._resource_map):
                raise IndexError(f'Index out of range: {key}')

            return self._resource_map[list(self._resource_map.keys())[key]]

        return self._resource_map[key]

    def __repr__(self):
        return f'MappableResource({", ".join(self._resource_map.keys())})'

@immutable
class _ResourceView(MappableResource[Any]):
    def __init__(
            self,
            resources: Mapping[str, WorkflowResource[Any]],
            task_name: str,
    ) -> None:
        super().__init__(resources)
        self.task_name = task_name if task_name else '<unknown>'

@immutable
class _PathView(MappableResource[Path]):
    def __init__(
            self,
            resources: Mapping[str, WorkflowResource[Path]],
            task_name: str,
    ) -> None:
        super().__init__(resources)
        self.task_name = task_name if task_name else '<unknown>'

@immutable
class ResourceContainer(Taggable, MappableResource[Any]):
    """Container for workflow tasks.

    :ivar task_name: Name of the task.
    """
    task_name: str = CheckedString(1, match=r'^[a-zA-Z0-9_]+$')
    _paths: dict[str, WorkflowResource[Path]]
    _resources: dict[str, WorkflowResource[Any]]
    _all_resources: Mapping[str, WorkflowResource[Any]]

    def __init__(
            self,
            task_name: str,
            paths: Optional[Iterable[WorkflowResource[Path]] | WorkflowResource[Any]] = None,
            resources: Optional[Iterable[WorkflowResource[Any]] | WorkflowResource[Any]] = None,
            tags: Optional[Mapping[str, str]] = None,
    ):
        """Initialize the task container.

        :param resources: Resources in this container.
        :param tags: Tags for the container.
        """
        super().__init__(tags)

        self.task_name = task_name
        self._paths = dict()
        self._resources = dict()

        if isinstance(paths, WorkflowResource):
            paths = (paths,)

        if isinstance(resources, WorkflowResource):
            resources = (resources,)

        for resource in (paths if paths is not None else ()):
            if resource.key in self._paths:
                raise ValueError(f'Duplicate path key: {resource.key}')
            self._paths[resource.key] = resource

        for resource in (resources if resources is not None else ()):
            if resource.key in self._resources:
                raise ValueError(f'Duplicate resource key: {resource.key}')
            self._resources[resource.key] = resource

        self._all_resources = ChainMap(self._paths, self._resources)

        MappableResource.__init__(self, self._all_resources)

        self.paths = _PathView(self._paths,self.task_name)
        self.resources = _ResourceView(self._resources, self.task_name)

    def __repr__(self) -> str:
        return f'WorkflowTaskResources({self.task_name}, Paths({', '.join(self._paths.keys())}), Resources({', '.join(self._resources.keys())}), {super().__repr__()})'


def _is_path_resource(
    resource: WorkflowResource[Any],
) -> TypeGuard[WorkflowResource[Path]]:
    """Check if a resource is a path resource.

    :param resource: Resource to check.

    :return: True if the resource is a path resource, False otherwise.
    """
    return isinstance(resource.value, Path)
