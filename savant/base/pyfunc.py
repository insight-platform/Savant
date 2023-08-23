"""PyFunc definitions."""
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import util as importlib_util, import_module, reload
from importlib.machinery import ModuleSpec
from types import ModuleType
from pathlib import Path
from typing import Any, Dict, Optional, Callable

import logging
from savant.gstreamer import Gst  # noqa: F401
from savant.utils.inotify_manager import INotifyManager


logger = logging.getLogger(__name__)


class PyFuncException(Exception):
    """PyFunc exception class."""


class PyFuncNoopCallException(PyFuncException):
    """PyFunc no-op call exception class."""


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
        self.logger = logging.getLogger(self.__module__)


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


class PyFuncNoopImpl(BasePyFuncPlugin, BasePyFuncCallableImpl):
    def process_buffer(self, buffer: Gst.Buffer):
        """Process a Gst buffer."""
        logger.debug('Noop pyfunc, process_buffer() called.')
        raise PyFuncNoopCallException('Called process_buffer() from noop pyfunc')

    def __call__(self, *args, **kwargs) -> Any:
        """Call self as a function."""
        logger.debug('Noop pyfunc, __call__() called.')
        raise PyFuncNoopCallException('Called __call__() from noop pyfunc')


@dataclass
class PyFunc:
    """PyFunc structure that defines instantiation parameters for an object
    implementing :py:class:`.BasePyFuncImpl`.

    To instantiate the specified module and class after configuration was finished,
    call :py:func:`load_user_code`.

    The instance will be available through :py:attr:`instance` property.

    A pyfunc can be configured to be run in dev mode. Pyfuncs in dev mode are
    reloaded at runtime before each call (each frame) if the source file has
    changed since the last one. Additionally, pyfuncs in dev mode gracefully
    recover from error in code, allowing the pipeline to stay up and continue processing.

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

    dev_mode: bool = False
    """Flag signifies if the pyfunc is in dev mode."""

    def __post_init__(self):
        self._module_instance = None
        self._instance = None
        self._callable = None
        self._load_complete: bool = False

    def load_user_code(self):
        """Load (or reload) the user module/class specified in the PyFunc fields.
        It's necessary to call this at least once before starting the pipeline.
        """
        if self._load_complete and not self.dev_mode:
            return
        load_ok = True
        logger.debug(
            'Loading user code start, pyfunc module %s, class %s, id %s',
            self.module,
            self.class_name,
            id(self),
        )
        # try to load module spec
        spec = None
        try:
            spec = pyfunc_module_spec_factory(self)
        except Exception as exc:
            if self.dev_mode:
                logger.exception('Error while getting module spec.')
                load_ok = False
            else:
                raise exc

        # try to set source file watch
        if self.dev_mode:
            if spec is None:
                logger.debug('No module spec, setting watch is impossible.')
            elif not spec.has_location:
                logger.debug(
                    'Module spec with undetermined location, setting watch is impossible.'
                )
            else:
                logger.debug('Setting watch for pyfunc module source.')
                INotifyManager().add_watch(spec.origin, id(self))

        # try to instatiate module
        module_instance = None
        if self.dev_mode:
            # if dev mode, it is possible to
            # leave module instance as None
            # need to reload instead of load
            if spec is None:
                logger.debug('No module spec, skip getting module instance.')
            elif self._module_instance:
                logger.debug('Module was previously loaded, doing reload.')
                try:
                    module_instance = reload_module(self._module_instance)
                except Exception as exc:
                    logger.exception('Error while reloading module instance.')
                    load_ok = False
                else:
                    # if no error, also cache new instance
                    self._module_instance = module_instance
            else:
                try:
                    module_instance = pyfunc_module_factory(spec)
                except Exception as exc:
                    logger.exception('Error while getting module instance.')
                    load_ok = False
        else:
            module_instance = pyfunc_module_factory(spec)

        # try to instantiate class
        pyfunc_impl_instance = None
        if self.dev_mode and module_instance is None:
            logger.debug('No module instance, skip getting class instance.')
        else:
            try:
                pyfunc_impl_instance = pyfunc_impl_factory(self, module_instance)
            except Exception as exc:
                if self.dev_mode:
                    logger.exception('Error getting class instance from module.')
                    load_ok = False
                else:
                    raise exc

        if self.dev_mode and pyfunc_impl_instance is None:
            pyfunc_impl_instance = PyFuncNoopImpl()
            logger.debug('No pyfunc impl, using noop placeholder.')

        self._instance = pyfunc_impl_instance
        self._callable = callable_factory(self._instance)

        if self.dev_mode and not self._load_complete:
            # if the module was loaded once
            # next attempts should be to reload a cached ModuleType instance
            self._module_instance = module_instance

        self._load_complete = True
        logger.debug(
            'Loading user code complete, pyfunc module %s, class %s, id %s',
            self.module,
            self.class_name,
            id(self),
        )
        return load_ok

    @property
    def instance(self) -> BasePyFuncImpl:
        """Returns resolved PyFunc implementation."""
        if self.dev_mode:
            self.check_reload()
        return self._instance

    def __call__(self, *args, **kwargs) -> Any:
        """Calls resolved PyFunc implementation if its a subclass of
        :py:class:`BasePyFuncCallableImpl`, otherwise no-op.
        """
        if self.dev_mode:
            self.check_reload()
        return self._callable(*args, **kwargs)

    def check_reload(self):
        """Checks if reload is needed and reloads in case it is."""

        if INotifyManager().is_changed(id(self)):
            logger.info(
                'Dev mode is enabled and changes in "%s.%s" are detected; reloading.'
                , self.module, self.class_name
            )
            load_ok = self.load_user_code()
            if load_ok:
                logger.info('The module "%s.%s" reloading complete: OK', self.module, self.class_name)
            else:
                logger.info('The module "%s.%s" reloading finsihed: Fail', self.module, self.class_name)
        else:
            logger.trace('Dev mode is enabled and no changes in "%s.%s" are detected; continuing.', self.module, self.class_name)


def reload_module(module: ModuleType) -> ModuleType:
    """Reload a previously loaded module.

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


def pyfunc_module_spec_factory(pyfunc: PyFunc) -> ModuleSpec:
    """Get a module spec for the module specified in the pyfunc."""
    if not getattr(pyfunc, 'module', None):
        raise PyFuncException('Python module name or path is required.')

    module_path = Path(pyfunc.module).resolve()
    if module_path.exists():
        spec = importlib_util.spec_from_file_location(module_path.stem, module_path)
    else:
        spec = importlib_util.find_spec(pyfunc.module, package=None)

    if not spec.has_location:
        # can be a built-in or a namespace packge, for example
        logger.warning(
            'Attempting to load a PyFunc with undetermined location. Is it really a user module? %r',
            pyfunc,
        )
    return spec


def pyfunc_module_factory(spec: ModuleSpec) -> ModuleType:
    """Instantiate a module from module spec."""
    module_instance = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module_instance)
    # reloading requires module being put into the modules dict
    # only do so after the loader.exec_module call as it can fail
    sys.modules[spec.name] = module_instance
    return module_instance


def pyfunc_impl_factory(pyfunc: PyFunc, module_instance: ModuleType) -> BasePyFuncImpl:
    """Instantiate the pyfunc class from module instance."""
    if not getattr(pyfunc, 'class_name', None):
        raise PyFuncException('Python class name is required.')

    pyfunc_class = getattr(module_instance, pyfunc.class_name)
    if pyfunc.kwargs:
        pyfunc_instance = pyfunc_class(**pyfunc.kwargs)
    else:
        pyfunc_instance = pyfunc_class()

    if not isinstance(pyfunc_instance, BasePyFuncImpl):
        raise PyFuncException(
            f'"{pyfunc_instance}" should be an instance of "BasePyFuncImpl" subclass.'
        )

    return pyfunc_instance


def callable_factory(pyfunc_impl: BasePyFuncImpl) -> Callable[..., Any]:
    """Get a callable from a pyfunc implementation."""
    if isinstance(pyfunc_impl, BasePyFuncCallableImpl):
        return pyfunc_impl
    return lambda *args, **kwargs: None
