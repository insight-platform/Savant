"""Registry utils."""
from typing import Any, Dict, Iterable, Iterator, Tuple


class Registry(Iterable[Tuple[str, Any]]):
    """The registry that provides name -> object mapping.

    To create a registry (e.g. a sink registry):

    .. code-block:: python

        SINK_REGISTRY = Registry('sink')

    To register an object:

    .. code-block:: python

        @SINK_REGISTRY.register('mysink')
        class MySink:
            ...

    Or:

    .. code-block:: python

        SINK_REGISTRY.register('mysink', MySink)
    """

    def __init__(self, name: str) -> None:
        """
        :param name: the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        assert name not in self._obj_map, (
            f'An object named "{name}" was already registered'
            f' in "{self._name}" registry.'
        )
        self._obj_map[name] = obj

    def register(self, name: str, obj: Any = None) -> Any:
        """Register the given object under the name.

        Can be used as a decorator.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: Any) -> Any:
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        self._do_register(name, obj)

    def get(self, name: str) -> Any:
        """Try to get name from registry."""
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry."
            )
        return ret

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __iter__(self) -> Iterator[str]:
        return iter(self._obj_map.keys())
