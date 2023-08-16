"""PyFunc definitions."""
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import util as importlib_util, import_module, reload
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable
from types import ModuleType
from savant.gstreamer import Gst  # noqa: F401
from savant.utils.inotify_manager import INotifyManager
from savant.utils.logging import get_logger

logger = get_logger(__name__)

class BasePyFuncImpl(ABC):
    """Base class for a PyFunc implementation. PyFunc implementations are
    defined in and instantiated by a :py:class:`.PyFunc` structure.

    :param kwargs: Custom keyword arguments.
        They will be available inside the class instance,
        as fields with the argument name.
    """

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.logger = get_logger(self.__module__)


class BasePyFuncPlugin(BasePyFuncImpl):
    """Base class for a PyFunc implementation to be used in a ``pyfunc``
    gstreamer element.

    PyFunc implementations are defined in and instantiated by a
    :py:class:`.PyFunc` structure.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gst_element: Optional[Gst.Element] = None

    def on_start(self) -> bool:
        """Do on plugin start."""
        return True

    def on_stop(self) -> bool:
        """Do on plugin stop."""
        return True

    def on_event(self, event: Gst.Event):
        """Do on event."""

    @abstractmethod
    def process_buffer(self, buffer: Gst.Buffer):
        """Process gstreamer buffer. Throws an exception if fatal error has
        occurred.

        :param buffer: Gstreamer buffer.
        """


class BasePyFuncCallableImpl(BasePyFuncImpl):
    """Base class for a PyFunc implementation as a callable.

    PyFunc implementations are defined in and instantiated by a
    :py:class:`.PyFunc` structure.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Call self as a function."""


@dataclass
class PyFunc:
    """PyFunc structure that defines instantiation parameters for an object
    implementing :py:class:`.BasePyFuncImpl`.

    Module and class will be resolved and imported
    with :py:func:`~savant.base.pyfunc.resolve_pyfunc`.
    Instantiation will be done in ``__post__init__()``
    and the instance will be available through :py:attr:`instance` property.

    .. note::

        This class is meant to be defined in a module config, not instantiated directly.

    .. note::

        Check :ref:`pipeline element hierarchy<pipeline_element_hierarchy>` to see which
        pipeline elements use PyFunc structure.

    For example, define a :py:class:`~savant.config.schema.PyFuncElement`

    .. code-block:: yaml

        - element: pyfunc
          module: module.pyfunc_implementation_module
          class_name: PyFuncImplementationClass

    where ``PyFuncImplementationClass`` inherits from :py:class:`.BasePyFuncPlugin` or
    from :py:class:`~savant.deepstream.pyfunc.NvDsPyFuncPlugin` for Deepstream pipeline.
    """

    module: str
    """Name to import or module path."""

    class_name: str
    """Python class name to instantiate."""

    kwargs: Optional[Dict[str, Any]] = None
    """Class initialization keyword arguments."""

    def __post_init__(self):
        module_instance = pyfunc_module_factory(self)
        self._instance: BasePyFuncImpl = pyfunc_impl_factory(self, module_instance)
        self._callable = callable_factory(self._instance)

    @property
    def instance(self) -> BasePyFuncImpl:
        """Returns resolved PyFunc implementation."""
        return self._instance

    def __call__(self, *args, **kwargs) -> Any:
        """Calls resolved PyFunc implementation if its a subclass of
        :py:class:`BasePyFuncCallableImpl`, otherwise no-op.
        """
        return self._callable(*args, **kwargs)

@dataclass
class PyFuncDynamicReloadable(PyFunc):

    def __post_init__(self):
        self._module_instance = pyfunc_module_factory(self)
        self._instance: BasePyFuncImpl = pyfunc_impl_factory(self, self._module_instance)
        self._callable = callable_factory(self._instance)
        INotifyManager().add_watch(self._module_instance.__file__, id(self))

    @property
    def instance(self) -> BasePyFuncImpl:
        """Returns resolved PyFunc implementation."""
        self._check_reload()
        return self._instance

    def __call__(self, *args, **kwargs) -> Any:
        """Calls resolved PyFunc implementation if its a subclass of
        :py:class:`BasePyFuncCallableImpl`, otherwise no-op.
        """
        self._check_reload()
        return self._callable(*args, **kwargs)

    def _check_reload(self):
        py_target = self.module, self.class_name
        if INotifyManager().is_changed(id(self)):
            logger.info('Pyfunc %s Reload %s', id(self), py_target)
            self._module_instance = reload_module(self._module_instance)
            self._instance: BasePyFuncImpl = pyfunc_impl_factory(self, self._module_instance)
            self._callable = callable_factory(self._instance)
        else:
            logger.info('Pyfunc %s Unchanged %s', id(self), py_target)

def reload_module(module: ModuleType) -> ModuleType:
    """

    .. note::

        When a module is reloaded, its dictionary (containing the module's global variables) is retained.
        Redefinitions of names will override the old definitions, so this is generally not a problem.
        If the new version of a module does not define a name that was defined by the old version, the old definition remains.

        https://docs.python.org/3/library/importlib.html#importlib.reload

        This is the reason for clearing module's attributes before reloading.

    """
    for attr in dir(module):
        if attr not in ('__name__', '__file__'):
            delattr(module, attr)
    return reload(module)

def pyfunc_factory(module, class_name, dynamic_reload, **kwargs) -> PyFunc:
    """Whether to reload the module and re-instantiate the class
    on changes detected in the source file.
    """
    if dynamic_reload:
        return PyFuncDynamicReloadable(module, class_name, **kwargs)
    return PyFunc(module, class_name, **kwargs)

def pyfunc_module_factory(pyfunc: PyFunc) -> ModuleType:
    assert pyfunc.module, 'Python module name or path is required.'
    assert pyfunc.class_name, 'Python class name is required.'

    module_path = Path(pyfunc.module).resolve()

    if module_path.exists():
        module_name = module_path.stem
        spec = importlib_util.spec_from_file_location(module_name, module_path)
        module_instance = importlib_util.module_from_spec(spec)
        sys.modules[module_name] = module_instance
        spec.loader.exec_module(module_instance)
    else:
        module_instance = import_module(pyfunc.module)

    return module_instance

def pyfunc_impl_factory(pyfunc: PyFunc, module_instance: ModuleType) -> BasePyFuncImpl:
    pyfunc_class = getattr(module_instance, pyfunc.class_name)
    if pyfunc.kwargs:
        pyfunc_instance = pyfunc_class(**pyfunc.kwargs)
    else:
        pyfunc_instance = pyfunc_class()

    assert isinstance(
        pyfunc_instance, BasePyFuncImpl
    ), f'"{pyfunc_instance}" should be an instance of "BasePyFuncImpl" subclass.'

    return pyfunc_instance

def callable_factory(pyfunc_impl: BasePyFuncImpl) -> Callable[...,Any]:
    if isinstance(pyfunc_impl, BasePyFuncCallableImpl):
        return pyfunc_impl
    return lambda *args, **kwargs: None
