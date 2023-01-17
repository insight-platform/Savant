"""Module and pipeline elements configuration templates."""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json
from omegaconf import MISSING, DictConfig, OmegaConf
from savant.base.pyfunc import PyFunc


@dataclass
class DynamicGstProperty:
    """Allows configuring a gstreamer element property to be automatically
    updated to current value of a dynamic parameter from parameter storage."""

    storage_key: str
    """Dynamic parameters name in the storage."""

    default: Any
    """Default value for the property."""


@dataclass
class PipelineElement:
    """Base pipeline element configuration template. Validates entries in a
    module config file under ``pipeline.source``, ``pipeline.elements`` and
    ``pipeline.sink``.

    Look for examples in :py:attr:`.element` documentation.
    """

    element: str = MISSING
    """Either a Gstreamer element name (gst factory name)
    or a short notation string to define :py:attr:`.element`, :py:attr:`.element_type`
    and :py:attr:`.version` at the same time.

    Short notation string format is ``<element>@<element_type>:<version>``

    .. note:: Version is ``v1`` by default.

    Examples.

    The following three element definitions are equivalent:

    .. code-block:: yaml

        - element: nvinfer
          element_type: attribute_model
          version: v1

        ...

        - element: nvinfer@attribute_model:v1

        ...

        - element: nvinfer@attribute_model

    Some elements might not have subtypes, in this case :py:attr:`.version` in the
    short notation can be defined immediately after the :py:attr:`.element`:

    .. code-block:: yaml

        - element: drawbin:v1
          location: /data/frames/image_%06d.jpg

    .. warning::

        Mixing short notation and full definition is not supported.

        Examples of unsupported notation mixing:

        .. code-block:: yaml

            - element: nvinfer@attribute_model
              version: v1

            ...

            - element: nvinfer:v1
              element_type: attribute_model

            ...

            - element: nvinfer
              element_type: attribute_model:v1
    """

    element_type: Optional[str] = None
    """Element type/subtype, can be defined as a substring of the :py:attr:`.element`.

    For example, ``detector`` in

    .. code-block:: yaml

        - element: nvinfer@detector
    """

    version: str = 'v1'
    """Element version, can be defined as a substring of the :py:attr:`.element`.

    For example, ``v1`` in

    .. code-block:: yaml

        - element: nvinfer@detector:v1
    """

    name: Optional[str] = None
    """GstElement instance name. Arbitrary string, useful for identifying
    pipeline elements.
    """

    properties: Dict[str, Any] = field(default_factory=dict)
    """GstElement properties."""

    dynamic_properties: Dict[str, DynamicGstProperty] = field(default_factory=dict)
    """GstElement properties that can be updated during runtime."""

    @property
    def full_name(self):
        """Full element name."""
        return get_element_name(self)


def get_element_name(element: Union[DictConfig, PipelineElement]) -> str:
    """Returns the full name of the element, including the element factory,
    type, and name. The function is needed to get a uniform element name for
    different element representations (DictConfig/PipelineElement).

    :param element: Element.
    :return: Element name.
    """
    if isinstance(element, PipelineElement):
        element_config = OmegaConf.structured(element)
    else:
        element_config = element

    full_name = element_config.element

    if element_config.get('element_type'):
        full_name += f'@{element_config.element_type}'

    if element_config.get('version'):
        full_name += f':{element_config.version}'

    if element_config.get('name'):
        full_name += f'(name={element_config.name})'

    return full_name


@dataclass
class DrawFunc(PyFunc):
    """A callable PyFunc that will use an object implementing
    :py:class:`~savant.deepstream.base_drawfunc.BaseNvDsDrawFunc`
    to draw metadata on frames.

    .. note::

        Default values for :py:attr:`.module` and :py:attr:`.class_name` attributes
        are set to use :py:class:`~savant.deepstream.drawfunc.NvDsDrawFunc` drawbin
        implementation.
    """

    module: str = 'savant.deepstream.drawfunc'
    """Module name to import."""

    class_name: str = 'NvDsDrawFunc'
    """Python class name to instantiate."""

    rendered_objects: Optional[Dict[str, Dict[str, Any]]] = None
    """Objects that will be rendered on the frame
    
    For example,

    .. code-block:: yaml
        - element: drawbin
            rendered_objects:
                tracker:
                    person: red
                yolov7obb:
                    person: green

    """

    kwargs: Optional[Dict[str, Any]] = None
    """Class initialization keyword arguments."""

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        self.kwargs.update({'rendered_objects': self.rendered_objects})
        super().__post_init__()


@dataclass
class PyFuncElement(PipelineElement, PyFunc):
    """A pipeline element that will use an object implementing
    :py:class:`~savant.base.pyfunc.BasePyFuncPlugin` to apply custom processing to
    gstreamer buffers.

    For example,

    .. code-block:: yaml

        - element: pyfunc
          module: module.pyfunc_implementation_module
          class_name: PyFuncImplementationClass
    """

    element: str = 'pyfunc'
    """``"pyfunc"`` is the fixed gstreamer element class for PyFuncElement."""

    def __post_init__(self):
        kwargs = {}
        if 'kwargs' in self.properties and self.properties['kwargs']:
            kwargs = json.loads(self.properties['kwargs'])
        if self.kwargs:
            kwargs.update(self.kwargs)

        if 'name' in self.kwargs:
            logging.warning("'name' is reserved name argument and will be replaced")
        if self.name:
            kwargs.update({'name': self.name})
        else:
            kwargs.update({'name': self.class_name})

        self.properties.update(
            {
                'module': self.module,
                'class': self.class_name,
                'kwargs': json.dumps(kwargs),
            }
        )


@dataclass
class ModelElement(PipelineElement):
    """A pipeline element that will run inference with specified deep learning
    model.

    For example,

    .. code-block:: yaml

        - element: nvinfer
          element_type: detector
          name: my_detector
          model:
              <model specification>
    """

    name: str = MISSING
    """GstElement instance name. Arbitrary string, useful for identifying
    pipeline elements. Mandatory for model elements.
    """

    model: Any = MISSING
    """Model configuration, any subclass of :py:class:`~savant.base.model.Model`.
    Check detailed Model hierarchy at links :ref:`base <model_hierarchy_base>`,
    :ref:`nvinfer <model_hierarchy_nvinfer>`.
    """


@dataclass
class Stage:
    """One level elements wrapper."""

    # TODO: Do we really need it? -> Implement/Remove

    name: Optional[str] = None
    """Stage name."""

    elements: List[PipelineElement] = MISSING
    """List of stage elements."""


@dataclass
class Pipeline:
    """Pipeline configuration template. Validates entries in a module config
    file under ``pipeline``. For example,

    .. code-block:: yaml

        pipeline:
            source:
                element: uridecodebin
                properties:
                    uri: file:///data/test.mp4
            elements:
                # user-defined pipeline elements
            sink:
                - element: console_sink
    """

    # TODO: Add format, e.g. NvDs

    source: PipelineElement = MISSING
    """The source element of a pipeline."""

    # elements: MISSING?  # Union[] is not supported -> Any
    elements: List[Any] = field(default_factory=list)
    """Pipeline's main contents: sequence of Pipe that implement all frame
    processing operations. Can be a :py:class:`PipelineElement` or any subclass of it.
    """

    draw_func: Optional[DrawFunc] = None

    sink: List[PipelineElement] = field(default_factory=list)
    """Sink elements of a pipeline."""


@dataclass
class Module:
    """Module configuration template. Top-level configuration entity, validates
    all entries in a module config file. For example,

    .. code-block:: yaml

        name: module_name
        parameters:
        pipeline:
            source:
                element: uridecodebin
                properties:
                    uri: file:///data/test.mp4
            elements:
            sink:
                - element: console_sink
    """

    name: str
    """Module name."""

    parameter_init_priority: Dict[str, int] = field(default_factory=dict)
    """Priority of use for various sources during module's parameters initialization
    through ``initializer`` resolver. Lower numbers mean higher priority. Example:

    .. code-block:: yaml

        parameter_init_priority:
            environment: 20
            etcd: 10

    Two init sources are configured, ``etcd`` (higher priority)
    and ``environment`` (lower priority).
    For every parameter that is configured to use ``initializer`` resolver, eg

    .. code-block:: yaml

        parameters:
            frame_width: ${initializer:frame_width,1280}

    Etcd storage will be polled for the current value first,
    in the event etcd is unavailable resolver will
    try to get ``frame_width`` environment variable, and if that is not set,
    then default value of 1280 will be used.
    """

    parameters: Dict[str, Any] = field(default_factory=dict)
    """Module parameters."""

    dynamic_parameters: Dict[str, Any] = field(default_factory=dict)
    """Those module parameters current value of which is dependent on remote storage
    and can be updated at runtime.

    For example, ``roi`` config node in

    .. code-block:: yaml

        dynamic_parameters:
            # (x_center, y_center, width, height, angle)
            roi: [960, 540, 1920, 1080, 0]
    """

    pipeline: Pipeline = MISSING
    """Pipeline configuration.
    """
