"""Base deep learning model configuration templates."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple
from omegaconf import MISSING
from savant.base.pyfunc import PyFunc
from savant.meta.constants import PRIMARY_OBJECT_LABEL
from savant.remote_file.schema import RemoteFile


class ModelPrecision(Enum):
    """Enum for model data format to be used by inference."""

    FP32 = 0
    """FP32 precision inference."""

    INT8 = 1
    """INT8 precision inference."""

    FP16 = 2
    """FP16 precision inference."""


class ModelColorFormat(Enum):
    """Enum for input image color format expected by the model."""

    RGB = 0
    """RGB input images."""

    BGR = 1
    """BGR input images."""

    GRAY = 2
    """Gray input images."""


@dataclass
class PreprocessObjectTensor:
    """Object image preprocessing function configuration."""

    custom_function: str = MISSING
    """Object image preprocessing function. It can be a Python function:

    1. Function should take one argument of type ``pysavantboost.Image``
       and return ``pysavantboost.Image``
    2. Function should be referenced in the form of ``module:function_name``

    or a C++ function

    1. Function should take one argument of type ``pysavantboost.Image``
       and return ``pysavantboost.Image``
    2. Function should be referenced in the form of ``library.so:function_name``
    """

    padding: Tuple[int, int] = (0, 0)
    """Setting X and Y padding (in pixels) around the object bbox allows specifying
    an extended image region that includes not just the object,
    but also its immediate surroundings.
    """


@dataclass
class ModelInput:
    """Model input parameters configuration template. Validates entries in a
    module config file under ``model.input``.

    Example,

    .. code-block:: yaml

        model:
            input:
                object: 'frame'
                shape: [3, 240, 240]
                maintain_aspect_ratio: True
    """

    object: str = PRIMARY_OBJECT_LABEL
    """A text label in the form of ``model_name.object_label``.
    Indicates objects that will be used as input data.
    Special value `frame` is used to specify the entire frame as model input.
    """

    # TODO: Add model input type
    #  format: ModelInputFormat = ModelInputFormat.Image

    # TODO: Specify image type-specific parameters
    #  order: ImageTensorOrder = ImageTensorOrder.NCHW
    #  ...

    layer_name: Optional[str] = None
    """Model input layer name.
    """

    shape: Optional[Tuple[int, int, int]] = None
    """``(Channels, Height, Width)`` tuple that indicates input image shape.

    Example

    .. code-block:: yaml

        shape: [3, 224, 224]
    """

    maintain_aspect_ratio: bool = False
    """Indicates whether the input preprocessing should maintain image aspect ratio.
    """

    # TODO: Add `symmetric-padding` support.

    # TODO: Enhance scaling options
    #  range: Tuple[] = (0, 255) or (0.0, 1.0)

    # TODO: Support per channel scaling, mean/std
    scale_factor: float = 1.0
    """Pixel scaling/normalization factor.
    """

    offsets: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Array of mean values of color components to be subtracted from each pixel.

    Example

    .. code-block:: yaml

        offset: [0.0, 0.0, 0.0]
    """

    color_format: ModelColorFormat = ModelColorFormat.RGB
    """Color format required by the model.

    Example

    .. code-block:: yaml

        color_format: rgb
        # color_format: bgr
        # color_format: gray
    """

    preprocess_object_meta: Optional[PyFunc] = None
    """Object metadata preprocessing.

    It should be defined using :py:class:`~savant.base.pyfunc.PyFunc`.

    Preprocessing implementation should be written as a subclass of
    :py:class:`~savant.base.input_preproc.BasePreprocessObjectMeta`.
    """

    preprocess_object_tensor: Optional[PreprocessObjectTensor] = None
    """Object image preprocessing Python/C++ function configuration.

    .. todo:: object_tensor_padding can be one of P, (PX, PY),
        (PLeft, PTop, PRight, PBottom), absolute (px) vs relative (%) padding unit

    """

    @property
    def height(self):
        """Input image height."""
        return self.shape[1]

    @property
    def width(self):
        """Input image width."""
        return self.shape[2]


@dataclass
class ModelOutput:
    """Model output parameters configuration template. Validates entries in a
    module config file under ``model.output``.

    .. code-block:: yaml

        model:
            output:
                layer_names: [output]
    """

    layer_names: List[str] = field(default_factory=list)
    """Specify model output layer names.
    """

    converter: Optional[PyFunc] = None
    """Model output converter. Converter is used to transform raw tensor
    output into Savant data format.

    Converter should be defined using :py:class:`~savant.base.pyfunc.PyFunc`.

    Converter implementation should be written as a subclass of
    :py:class:`~savant.base.converter.BaseObjectModelOutputConverter` or
    :py:class:`~savant.base.converter.BaseAttributeModelOutputConverter` or
    :py:class:`~savant.base.converter.BaseComplexModelOutputConverter`
    depending on the model type.
    """


@dataclass
class ObjectModelOutputObject:
    """ObjectModel output objects configuration template. Validates entries in
    a module config under ``model.output.objects``.

    Example,

    .. code-block: yaml

        objects:
            - class_id: 0
              label: cat
            - class_id: 1
              label: dog
    """

    class_id: int
    """Integer class id from object model inference results.
    """

    label: str
    """Text label for the object class id.
    Can subsequently be used in other model's input configuration.
    """

    selector: Optional[PyFunc] = None
    """Model output selector.

    Selector should be defined using :py:class:`~savant.base.pyfunc.PyFunc`.

    Selector implementation should be written as a subclass of
    :py:class:`~savant.base.selector.BaseSelector`.
    """


@dataclass
class ObjectModelOutput(ModelOutput):
    """ObjectModel output configuration template.

    Validates entries in a module config file under ``model.output``.

    Example:

    .. code-block:: yaml

        model:
            # model configuration
            output:
                layer_names: [output]
                objects:
                    # output objects configuration
    """

    objects: List[ObjectModelOutputObject] = MISSING
    """Configuration for each output object class of an object model."""


@dataclass
class AttributeModelOutputAttribute:
    """AttributeModel output attribute configuration template. Validates
    entries in a module config under ``model.output.attributes``.

    Example,

    .. code-block:: yaml

        attributes:
            - name: color
              labels: ['red', 'green']
    """

    name: str = MISSING
    """Attribute name will be used in the label under which a model's results are added
    to metadata.
    """

    labels: List[str] = field(default_factory=list)
    """A list of text labels that correspond to class ids of a classification model.
    The number and order of these labels should match those of a classification model's
    result classes.
    """

    threshold: Optional[float] = None
    """Minimum threshold label probability.
    The model outputs the label having the highest probability
    if it is greater than this threshold."""

    multi_label: bool = False
    """Output all labels/values whose probability/confidence exceeds
    the :py:attr:`.threshold` value."""

    internal: bool = False
    """Indicates whether this attribute should be excluded from outbound messages.
    Set to ``True`` for attributes which are only useful for
    internal pipeline operation.

    .. todo:: can be specified for an object
    """


@dataclass
class AttributeModelOutput(ModelOutput):
    """AttributeModel output configuration template.

    Validates entries in a module config file under ``model.output``.

    Example:

    .. code-block:: yaml

        model:
            # model configuration
            output:
                layer_names: [output]
                attributes:
                    - name: cat_or_dog
                      threshold: 0.5
    """

    attributes: List[AttributeModelOutputAttribute] = MISSING
    """Configuration for each output attribute of an attribute model."""


@dataclass
class ComplexModelOutput(ObjectModelOutput, AttributeModelOutput):
    """ComplexModel output configuration template.

    Validates entries in a module config file under ``model.output``.

    Look for examples in :py:class:`.ObjectModelOutput` and
    :py:class:`.AttributeModelOutput`.
    """

    converter: PyFunc = MISSING
    """Model output converter is required for complex model.
    Converter is used to transform raw tensor output into Savant data format.
    """


@dataclass
class Model:
    """Base model configuration template.

    Validates entries in a module config file under ``element.model``.
    """

    local_path: Optional[str] = None
    """Path where all the necessary model files are placed.
    By default, the value of module parameter "model_path" and element name
    will be used ("model_path / element_name").
    """

    remote: Optional[RemoteFile] = None
    """Configuration of model files remote location.
    Supported schemes: s3, http, https, ftp.
    """

    model_file: Optional[str] = None
    """The model file, eg `yolov4.onnx`.

    .. note::

        The model file is specified without a location.
        The absolute path to the model file will be defined
        as ":py:attr:`.local_path`/:py:attr:`.model_file`".
    """

    batch_size: int = 1
    """Number of frames or objects to be inferred together in a batch.

    .. note:: In case the model is an NvInferModel and it is configured to
       use the TRT engine file directly, the default value for ``batch_size``
       will be taken from the engine file name, by parsing it according to the scheme
       {model_name}_b{batch_size}_gpu{gpu_id}_{precision}.engine
    """

    precision: ModelPrecision = ModelPrecision.FP16
    """Data format to be used by inference.

    Example

    .. code-block:: yaml

        precision: fp16
        # precision: int8
        # precision: fp32

    .. note:: In case the model is an NvInferModel and it is configured to
       use the TRT engine file directly, the default value for ``precision``
       will be taken from the engine file name, by parsing it according to the scheme
       {model_name}_b{batch_size}_gpu{gpu_id}_{precision}.engine
    """

    input: ModelInput = ModelInput()
    """Optional configuration of input data and custom preprocessing methods
    for a model. If not set, then input will default to entire frame.
    """


@dataclass
class ObjectModel(Model):
    """Object model configuration template. Validates entries in a module
    config file under ``element.model``.

    Use to configure a detector.
    """

    output: ObjectModelOutput = ObjectModelOutput()
    """Configuration for post-processing of an object model's results."""


@dataclass
class AttributeModel(Model):
    """Attribute model configuration template. Validates entries in a module
    config file under ``element.model``.

    Use to configure a classifier or ReID model.
    """

    output: AttributeModelOutput = AttributeModelOutput()
    """Configuration for post-processing of an attribute model's results."""


@dataclass
class ComplexModel(Model):
    """Complex model configuration template. Validates entries in a module
    config file under ``element.model``.

    Complex model combines object and attribute models, for example face
    detector that produces bounding boxes and landmarks.
    """

    output: ComplexModelOutput = ComplexModelOutput()
    """Configuration for post-processing of a complex model's results."""
