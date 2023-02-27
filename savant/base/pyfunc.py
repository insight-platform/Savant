"""PyFunc definitions."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import util as importlib_util, import_module
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from savant.gstreamer import Gst  # noqa: F401


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
        self._instance: BasePyFuncImpl = resolve_pyfunc(self)
        self._callable = (
            self._instance
            if isinstance(self._instance, BasePyFuncCallableImpl)
            else lambda *args, **kwargs: None
        )

    @property
    def instance(self) -> BasePyFuncImpl:
        """Returns resolved PyFunc implementation."""
        return self._instance

    def __call__(self, *args, **kwargs) -> Any:
        """Calls resolved PyFunc implementation if its a subclass of
        :py:class:`BasePyFuncCallableImpl`, otherwise no-op.
        """
        return self._callable(*args, **kwargs)


def resolve_pyfunc(pyfunc: PyFunc) -> BasePyFuncImpl:
    """Resolves PyFunc. Takes PyFunc definition and returns PyFunc
    implementation.

    :param pyfunc: PyFunc structure.
    :return: PyFunc implementation object.
    """
    assert pyfunc.module, 'Python module name or path is required.'
    assert pyfunc.class_name, 'Python class name is required.'

    module_path = Path(pyfunc.module).resolve()

    if module_path.exists():
        spec = importlib_util.spec_from_file_location(module_path.stem, module_path)
        module_instance = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module_instance)
    else:
        module_instance = import_module(pyfunc.module)

    pyfunc_class = getattr(module_instance, pyfunc.class_name)
    if pyfunc.kwargs:
        pyfunc_instance = pyfunc_class(**pyfunc.kwargs)
    else:
        pyfunc_instance = pyfunc_class()

    assert isinstance(
        pyfunc_instance, BasePyFuncImpl
    ), f'"{pyfunc_instance}" should be an instance of "BasePyFuncImpl" subclass.'

    return pyfunc_instance
