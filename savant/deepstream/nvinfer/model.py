"""Gst-nvinfer model configuration templates."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from omegaconf import MISSING
from savant.base.pyfunc import PyFunc
from savant.base.model import (
    Model,
    ModelInput,
    ObjectModel,
    ObjectModelOutput,
    ObjectModelOutputObject,
    AttributeModel,
    ComplexModelOutput,
    ComplexModel,
)
from savant.utils.registry import Registry

NVINFER_MODEL_TYPE_REGISTRY = Registry('nvinfer_model_type')


class NvInferModelFormat(Enum):
    """Enum for format of the provided model file."""

    CAFFE = 0
    """Caffe model."""

    UFF = 1
    """UFF model."""

    ONNX = 2
    """ONNX model."""

    ETLT = 3
    """Nvidia TAO model."""

    CUSTOM = 4  # YOLO?
    """Other format."""


class NvInferModelType(Enum):
    """Enum for type of the model (``network-type`` in `nvinfer`
    configuration)."""

    DETECTOR = 0
    """Detector model."""

    CLASSIFIER = 1
    """Classification model."""

    SEGMENTATION = 2
    """Semantic segmentation model."""

    INSTANCE_SEGMENTATION = 3
    """Instance segmentation model."""

    CUSTOM = 100
    """Custom model."""


@dataclass
class NvInferModelInput(ModelInput):
    """`nvinfer` model input configuration template.

    Example

    .. code-block:: yaml

        model:
            # model configuration
            input:
                layer_name: input_1
                shape: [3, 544, 960]
                scale_factor: 0.0039215697906911373
    """

    object_min_width: Optional[int] = None
    """Model infers only on objects with this minimum width."""

    object_min_height: Optional[int] = None
    """Model infers only on objects with this minimum height."""

    object_max_width: Optional[int] = None
    """Model infers only on objects with this maximum width."""

    object_max_height: Optional[int] = None
    """Model infers only on objects with this maximum height."""


@dataclass
class NvInferModel(Model):
    """Base configuration template for a `nvinfer` model."""

    input: NvInferModelInput = NvInferModelInput()
    """Optional configuration of input data and custom preprocessing methods
    for a model. If not set, then input will default to entire frame.
    """

    format: NvInferModelFormat = MISSING
    """Model file format.

    Example

    .. code-block:: yaml

        format: onnx
        # format: caffe
        # etc.
        # look in enum for full list of format options
    """

    config_file: Optional[str] = None
    """Model configuration file. Should be specified without a location."""

    int8_calib_file: Optional[str] = None
    """INT8 calibration file for dynamic range adjustment with an FP32 model.
    Required only for models in INT8."""

    engine_file: Optional[str] = None
    """Serialized model engine file."""

    proto_file: Optional[str] = None
    """Caffe model prototxt file.
    By default, the model file name (:py:attr:`.model_file`) will be used
    with the extension ``.prototxt``.
    """

    custom_config_file: Optional[str] = None
    """Configuration file for custom model, eg for YOLO.
    By default, the model file name (:py:attr:`.model_file`) will be used
    with the extension ``.cfg``.
    """

    mean_file: Optional[str] = None
    """Pathname of mean data file in PPM format."""

    label_file: Optional[str] = None
    """Pathname of a text file containing the labels for the model."""

    tlt_model_key: Optional[str] = None
    """Key for the TAO toolkit encoded model."""

    gpu_id: int = 0
    """Device ID of GPU to use for pre-processing/inference (dGPU only).

    .. note:: In case the model is configured to
       use the TRT engine file directly, the default value for ``gpu_id``
       will be taken from the :py:attr:`.engine_file`, by parsing it
       according to the scheme {model_name}_b{batch_size}_gpu{gpu_id}_{precision}.engine
    """

    # TODO: Add support for custom models.
    #  Currently it is supported for regular detector and classifier only.
    interval: int = 0
    """Specifies the number of consecutive batches to be skipped for inference."""

    # TODO: Add support for Gst-nvinfer props
    # symmetric-padding (with maintain-aspect-ratio) -
    #     Indicates whether to pad image symmetrically while scaling input.
    #     DeepStream pads the images asymmetrically by default.
    # workspace-size - Workspace size to be used by the engine, in MB
    # network-input-order - Order of the network input layer

    custom_lib_path: Optional[str] = None
    """Absolute pathname of a library containing custom method implementations
    for custom models."""

    engine_create_func_name: Optional[str] = None
    """Name of the custom TensorRT CudaEngine creation function."""


NVINFER_DEFAULT_OBJECT_SELECTOR = PyFunc(
    module='savant.selector',
    class_name='BBoxSelector',
    kwargs=dict(confidence_threshold=0.5, nms_iou_threshold=0.5),
)


@dataclass
class NvInferObjectModelOutputObject(ObjectModelOutputObject):
    """NvInferObjectModel output objects configuration template.."""

    selector: PyFunc = NVINFER_DEFAULT_OBJECT_SELECTOR
    """Model output selector."""


@dataclass
class NvInferObjectModelOutput(ObjectModelOutput):
    """NvInferObjectModel output configuration template for detector with
    aligned bboxes.

    Example

    .. code-block:: yaml

        model:
            # model configuration
            output:
                num_detected_classes: 4
                layer_names: [output]
                objects:
                    # output objects configuration
    """

    objects: Optional[List[NvInferObjectModelOutputObject]] = None
    """Configuration for each output object class of an object model."""

    num_detected_classes: Optional[int] = None
    """Number of classes detected by the model. Required for regular detector."""


@dataclass
class NvInferComplexModelOutput(ComplexModelOutput, NvInferObjectModelOutput):
    """ComplexModel output configuration template."""


@NVINFER_MODEL_TYPE_REGISTRY.register('detector')
@dataclass
class NvInferDetector(NvInferModel, ObjectModel):
    """Standard detector with orthogonal bboxes configuration template.

    Example

    .. code-block:: yaml

        - element: nvinfer@detector
          name: Primary_Detector
          model:
            format: caffe
            model_file: resnet10.caffemodel
            batch_size: 1
            precision: int8
            int8_calib_file: cal_trt.bin
            label_file: labels.txt
            input:
              scale_factor: 0.0039215697906911373
            output:
              num_detected_classes: 4
              layer_names: [conv2d_bbox, conv2d_cov/Sigmoid]
    """

    parse_bbox_func_name: Optional[str] = None
    """Name of the custom bounding box parsing function.
    If not specified, Gst-nvinfer uses the internal function
    for the resnet model provided by the SDK."""

    output: NvInferObjectModelOutput = NvInferObjectModelOutput()
    """Results post-processing configuration."""


@NVINFER_MODEL_TYPE_REGISTRY.register('attribute_model')
@NVINFER_MODEL_TYPE_REGISTRY.register('classifier')
@dataclass
class NvInferAttributeModel(NvInferModel, AttributeModel):
    """NvInferAttribute model configuration template.

    Use to configure classifiers, etc.

    Example

    .. code-block:: yaml

        - element: nvinfer@classifier
          name: Secondary_CarColor
          model:
            format: caffe
            model_file: resnet18.caffemodel
            mean_file: mean.ppm
            label_file: labels.txt
            precision: int8
            int8_calib_file: cal_trt.bin
            batch_size: 16
            input:
              object: Primary_Detector.Car
              object_min_width: 64
              object_min_height: 64
              color_format: bgr
            output:
              layer_names: [predictions/Softmax]
              attributes:
                - name: car_color
                  threshold: 0.51
    """

    parse_classifier_func_name: Optional[str] = None
    """Name of the custom classifier output parsing function.
    If not specified, Gst-nvinfer uses the internal parsing function
    for softmax layers."""


@NVINFER_MODEL_TYPE_REGISTRY.register('complex_model')
@dataclass
class NvInferComplexModel(NvInferModel, ComplexModel):
    """NvInferComplexModel configuration template.

    Complex model combines object and attribute models.

    For example face detector that produces bounding boxes and landmarks:

    .. code-block:: yaml

        - element: nvinfer@complex_model
          name: face_detector
          model:
            format: onnx
            config_file: config.txt
            output:
              layer_names: ['bboxes', 'scores', 'landmarks']
              converter:
                module: module.face_detector_coverter
                class_name: FaceDetectorConverter
              objects:
                - class_id: 0
                  label: face
                  selector:
                    module: savant.selector
                    class_name: BBoxSelector
                    kwargs:
                      confidence_threshold: 0.5
                      nms_iou_threshold: 0.5
              attributes:
                - name: landmarks
    """

    output: NvInferComplexModelOutput = NvInferComplexModelOutput()
    """Configuration for post-processing of a complex model's results."""


@NVINFER_MODEL_TYPE_REGISTRY.register('instance_segmentation')
@dataclass
class NvInferInstanceSegmentation(NvInferComplexModel):
    """Instance segmentation model configuration template."""

    parse_bbox_instance_mask_func_name: Optional[str] = None
    """Name of the custom instance segmentation parsing function.
    It is mandatory for instance segmentation network
    as there is no internal function."""
