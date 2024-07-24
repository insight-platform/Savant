"""`nvinfer` element configuration."""

import logging
import os.path
from collections import defaultdict
from pathlib import Path
from typing import Optional, Type

from omegaconf import DictConfig, OmegaConf
from savant_rs.utils.symbol_mapper import (
    RegistrationPolicy,
    get_object_id,
    parse_compound_key,
    register_model_objects,
)

from savant.base.model import (
    AttributeModel,
    ComplexModel,
    ModelColorFormat,
    ModelPrecision,
    ObjectModel,
)
from savant.config.schema import get_element_name
from savant.deepstream.nvinfer.file_config import NvInferConfig, NvInferConfigType
from savant.deepstream.nvinfer.model import (
    NVINFER_MODEL_TYPE_REGISTRY,
    NvInferInstanceSegmentation,
    NvInferModel,
    NvInferModelFormat,
    NvInferModelType,
    NvInferObjectModelOutputObject,
)
from savant.parameter_storage import param_storage
from savant.remote_file import process_remote
from savant.utils.logging import get_logger

__all__ = ['nvinfer_element_configurator', 'MERGED_CLASSES']

MERGED_CLASSES = defaultdict(dict)


class NvInferConfigException(Exception):
    """NvInfer config exception class."""


class NvTrackerConfigException(Exception):
    """NvTracker config exception class."""


def recognize_format_by_file_name(model_file: str):
    """Recognize model format by model file name."""
    if model_file.endswith('.onnx'):
        return NvInferModelFormat.ONNX
    if model_file.endswith('.uff'):
        return NvInferModelFormat.UFF
    if model_file.endswith('.etlt'):
        return NvInferModelFormat.ETLT
    if model_file.endswith('.caffemodel'):
        return NvInferModelFormat.CAFFE
    return NvInferModelFormat.CUSTOM


def nvinfer_element_configurator(
    element_config: DictConfig, module_config: DictConfig
) -> DictConfig:
    """Configure nvinfer element.

    :param element_config: Element configuration
    :param module_config: Module configuration
    :return: Complete and validated element configuration
    """

    # patch logger to get element name in message
    element_name = get_element_name(element_config)

    class _LoggerAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return f'Element {element_name}: {msg}', kwargs

    logger = _LoggerAdapter(get_logger(__name__), dict(element_name=element_name))

    logger.trace('Configuring nvinfer element %s', element_config)
    # check model type
    try:
        model_type: Type[NvInferModel] = NVINFER_MODEL_TYPE_REGISTRY.get(
            element_config.element_type
        )
        model = model_type()
    except KeyError as exc:
        raise NvInferConfigException(
            f'Invalid model type "{element_config.element_type}", '
            f'expected one of {list(NVINFER_MODEL_TYPE_REGISTRY)}.'
        ) from exc

    # check model specification
    model_config = element_config.get('model')
    if not model_config:
        raise NvInferConfigException('Model specification required.')

    # prepare parameters with case-insensitive values (enums)
    enum_params = {
        'format': NvInferModelFormat,
        'precision': ModelPrecision,
        'input.color_format': ModelColorFormat,
    }
    for param_name, enum in enum_params.items():
        cfg = model_config
        prm_name = param_name
        while '.' in prm_name and cfg is not None:
            section, prm_name = prm_name.split('.', 1)
            cfg = cfg.get(section)
        if cfg is None:
            continue
        if prm_name in cfg and cfg[prm_name]:
            try:
                cfg[prm_name] = enum[str(cfg[prm_name]).upper()]
            except KeyError as exc:
                raise NvInferConfigException(
                    f'Invalid value "{cfg[prm_name]}" for "module.{param_name}", '
                    f'expected one of {[value.name for value in enum]}.'
                ) from exc

    # setup path for the model files
    if not model_config.get('local_path'):
        model_config.local_path = str(
            Path(param_storage()['model_path']) / element_config.name
        )
        logger.info(
            'Path to the model files has been set to "%s".', model_config.local_path
        )
    model_path = Path(model_config.local_path)

    if model_config.get('remote'):
        download_path = Path(param_storage()['download_path']) / element_config.name
        process_remote(model_config.remote, download_path, model_path)

    # try to load nvinfer config
    nvinfer_config: Optional[NvInferConfigType] = None
    nvinfer_config_file = model_config.get('config_file')
    if nvinfer_config_file:
        config_file_path = model_path / nvinfer_config_file
        logger.debug('Trying to load nvinfer config file %s', config_file_path)
        if config_file_path.is_file():
            nvinfer_config = NvInferConfig.read_file(config_file_path)
            logger.info(
                'Configuration file "%s" has been loaded. '
                'Processing of the model will be done according '
                'to the given configuration file.',
                config_file_path,
            )
            model = NvInferConfig.to_model(nvinfer_config, model)
        else:
            raise NvInferConfigException(
                f'Configuration file "{config_file_path}" not found.'
            )

    logger.trace(
        'Merging model with model config\nmodel %s\nmodel config %s',
        model,
        model_config,
    )
    model_config_original = model_config
    model_config = OmegaConf.merge(model, model_config)
    logger.trace('Merging complete, result\n%s', model_config)

    # try to parse engine file and check for a match
    if model_config.engine_file:
        parse_result = NvInferConfig.parse_model_engine_file(model_config.engine_file)
        if parse_result:
            # if engine options are not set explicitly
            # get their values from engine file name
            if (
                not hasattr(model_config_original, 'batch_size')
                or model_config_original.batch_size is None
            ):
                model_config.batch_size = parse_result['batch_size']
                logger.debug(
                    'Model batch size is taken from engine file name and set to %d',
                    parse_result['batch_size'],
                )
            if (
                not hasattr(model_config_original, 'gpu_id')
                or model_config_original.gpu_id is None
            ):
                model_config.gpu_id = parse_result['gpu_id']
                logger.debug(
                    'Model gpu_id is taken from engine file name and set to %d',
                    parse_result['gpu_id'],
                )
            if (
                not hasattr(model_config_original, 'precision')
                or model_config_original.precision is None
            ):
                model_config.precision = parse_result['precision']
                logger.debug(
                    'Model precision is taken from engine file name and set to %s',
                    parse_result['precision'].name,
                )

            if (
                model_config.batch_size,
                model_config.gpu_id,
                model_config.precision,
            ) != (
                parse_result['batch_size'],
                parse_result['gpu_id'],
                parse_result['precision'],
            ):
                logger.info(
                    'Specified engine file "%s" does not match configuration: '
                    'batch_size=%d, gpu_id=%d, precision=%s.',
                    model_config.engine_file,
                    model_config.batch_size,
                    model_config.gpu_id,
                    model_config.precision.name,
                )
                model_config.engine_file = None

    # model or engine file must be specified
    model_file_required = True
    if model_config.engine_file:
        engine_file_path = model_path / model_config.engine_file
        if engine_file_path.is_file():
            model_file_required = False
        else:
            logger.warning('Model engine file "%s" not found.', engine_file_path)

    if model_config.model_file:
        model_file_path = model_path / model_config.model_file
        if not model_file_path.is_file():
            model_config.model_file = None
            if model_file_required:
                raise NvInferConfigException(
                    f'Model file "{model_file_path}" not found.'
                )
    else:
        if model_file_required:
            raise NvInferConfigException(
                'Model file (model.model_file) or '
                'engine file (model.engine_file) required.'
            )

    # generate model-engine-file if not set
    if not model_config.engine_file:
        device_id = None
        if model_config.enable_dla:
            device_id = f'dla{model_config.use_dla_core}'
        else:
            device_id = f'gpu{model_config.gpu_id}'
        model_config.engine_file = NvInferConfig.generate_model_engine_file(
            model_config.model_file,
            model_config.batch_size,
            device_id,
            model_config.precision,
        )
        logger.info('Model engine file has been set to "%s".', model_config.engine_file)

    # check model format-specific parameters required to build the engine
    if model_file_required:
        if not model_config.format:
            model_config.format = recognize_format_by_file_name(model_config.model_file)

        if model_config.format == NvInferModelFormat.CAFFE:
            if not model_config.proto_file:
                model_config.proto_file = Path(model_config.model_file).with_suffix(
                    '.prototxt'
                )
                logger.warning(
                    'Caffe model prototxt file has been set to "%s".',
                    model_config.proto_file,
                )
            proto_file_path = model_path / model_config.proto_file
            if not proto_file_path.is_file():
                raise NvInferConfigException(
                    f'Caffe model prototxt file "{proto_file_path}" not found.'
                )

        elif model_config.format == NvInferModelFormat.CUSTOM:
            if not model_config.custom_config_file:
                model_config.custom_config_file = Path(
                    model_config.model_file
                ).with_suffix('.cfg')
                logger.warning(
                    'Custom model configuration file has been set to "%s".',
                    model_config.custom_config_file,
                )
            custom_config_file_path = model_path / model_config.custom_config_file
            if not custom_config_file_path.is_file():
                raise NvInferConfigException(
                    f'Custom model cfg file "{custom_config_file_path}" not found.'
                )
            # use abs path for custom config file
            model_config.custom_config_file = str(custom_config_file_path.resolve())

            if model_config.custom_lib_path is None:
                raise NvInferConfigException('model.custom_lib_path is required.')
            try:
                lib_path = Path(model_config.custom_lib_path)
                if not lib_path.is_file():
                    raise NvInferConfigException(
                        f'model.custom_lib_path "{lib_path}" not found.'
                    )
            except TypeError as exception:
                raise NvInferConfigException(
                    f'model.custom_lib_path "{model_config.custom_lib_path}"'
                    ' is invalid.'
                ) from exception

            if model_config.engine_create_func_name is None:
                raise NvInferConfigException(
                    'model.engine_create_func_name is required.'
                )

        elif model_config.format == NvInferModelFormat.ETLT:
            if not model_config.tlt_model_key:
                model_config.tlt_model_key = 'tlt_encode'  # or 'nvidia_tlt'
                logger.warning(
                    'Key for the TAO encoded model (model.tlt_model_key) '
                    'has been set to "%s".',
                    model_config.tlt_model_key,
                )

        # UFF model requirements (some ETLT models are UFF originally, e.g. peoplenet)
        if model_config.format in (NvInferModelFormat.UFF, NvInferModelFormat.ETLT):
            if not model_config.input.layer_name:
                raise NvInferConfigException(
                    'Model input layer name (model.input.layer_name) required.'
                )
            if not model_config.input.shape:
                raise NvInferConfigException(
                    'Model input shape (model.input.shape) required.'
                )

        if model_config.format in (
            NvInferModelFormat.CAFFE,
            NvInferModelFormat.UFF,
            NvInferModelFormat.ETLT,
        ):
            if not model_config.output.layer_names:
                raise NvInferConfigException(
                    'Model output layer names (model.output.layer_names) required.'
                )

        # calibration file is required to build model in INT8
        if model_config.precision == ModelPrecision.INT8:
            if not model_config.int8_calib_file:
                raise NvInferConfigException(
                    'INT8 calibration file (model.int8_calib_file) required.'
                )
            int8_calib_file_path = model_path / model_config.int8_calib_file
            if not int8_calib_file_path.is_file():
                raise NvInferConfigException(
                    f'INT8 calibration file "{int8_calib_file_path}" not found.'
                )

    if model_config.output.converter:
        logger.info('Model output converter will be used.')

        # input shape is used in some converters,
        # e.g. to scale the output of yolo detector
        if not model_config.input.shape:
            raise NvInferConfigException(
                'Model input shape (model.input.shape) required.'
            )

        # output layer names are required to properly order the output tensors
        # for passing to the converter
        if not model_config.output.layer_names:
            raise NvInferConfigException(
                'Model output layer names (model.output.layer_names) required.'
            )

    # model type-specific parameters
    if issubclass(model_type, ObjectModel):
        # model_config.output.objects is mandatory for object models,
        # but it may be autogenerated based on labelfile or num_detected_classes

        label_file = model_config.get(
            'label_file',
            (
                nvinfer_config['property'].get('labelfile-path')
                if nvinfer_config
                else None
            ),
        )
        if model_config.output.objects:
            # highest priority is using manually defined model_config.output.objects
            if label_file:
                logger.warning(
                    'Model output objects labels are defined manually '
                    'and will be used instead of labels from "%s".',
                    label_file,
                )

        elif label_file:
            # try to load model object labels from file
            label_file_path = model_path / label_file
            if label_file_path.is_file():
                with open(label_file_path.resolve(), encoding='utf8') as file_obj:
                    model_config.output.objects = [
                        NvInferObjectModelOutputObject(class_id=class_id, label=label)
                        for class_id, label in enumerate(file_obj.read().splitlines())
                    ]
                if model_config.output.num_detected_classes:
                    logger.warning(
                        'Ignoring manually set value for '
                        '(model_config.output.num_detected_classes) '
                        'because labelfile is used.'
                    )
                model_config.output.num_detected_classes = len(
                    model_config.output.objects
                )
                logger.info(
                    'Model object labels have been loaded from "%s".',
                    label_file_path,
                )
        elif model_config.output.num_detected_classes:
            # generate labels (enumerate)
            model_config.output.objects = [
                NvInferObjectModelOutputObject(class_id=class_id, label=str(class_id))
                for class_id in range(model_config.output.num_detected_classes)
            ]
            logger.info(
                'Model object labels are not specified, '
                'object class identifiers will be used instead.',
            )

        if not model_config.output.objects:
            raise NvInferConfigException(
                'Model output objects config (model.output.objects) required.'
            )

        if (
            not model_config.output.converter
            and not model_config.output.num_detected_classes
        ):
            raise NvInferConfigException(
                'Number of detected classes (model.output.num_detected_classes) '
                'must be specified for regular detectors.'
            )

    # prepare and save resulting model config file
    nvinfer_config = NvInferConfig.from_model(model_config, nvinfer_config)

    # set model processing mode to "secondary" (frame is the only primary object)
    nvinfer_config['property']['process-mode'] = 2

    # if there is no input.object (model_uid, class_id) in the registry -
    # register and expect it in module input meta (external meta)
    model_name, label = parse_compound_key(model_config.input.object)
    parent_model_uid, parent_class_id = get_object_id(model_name, label)
    nvinfer_config['property']['operate-on-gie-id'] = parent_model_uid
    nvinfer_config['property']['operate-on-class-ids'] = parent_class_id

    # register the model
    if issubclass(model_type, (ObjectModel, ComplexModel)):
        output_objects = model_config.output.get('objects', [])
        if output_objects:
            logger.debug('Registering output objects for the model ...')
        for obj in output_objects:
            try:
                model_uid = register_model_objects(
                    element_config.name,
                    {obj.class_id: obj.label},
                    RegistrationPolicy.ErrorIfNonUnique,
                )
                logger.debug(
                    'Object id %s "%s" was registered.', obj.class_id, obj.label
                )
            except ValueError:
                _, obj_id = get_object_id(element_config.name, obj.label)
                logger.debug(
                    'Object label "%s" already registered for id %s. Merging id %s into id %s.',
                    obj.label,
                    obj_id,
                    obj.class_id,
                    obj_id,
                )
                MERGED_CLASSES[element_config.name][obj.class_id] = obj_id
    else:
        model_uid = register_model_objects(
            element_config.name,
            {},
            RegistrationPolicy.ErrorIfNonUnique,
        )
    nvinfer_config['property']['gie-unique-id'] = model_uid

    # set network type to custom if converter is specified for model output
    if model_config.output.converter:
        nvinfer_config['property']['output-tensor-meta'] = 1
        nvinfer_config['property']['network-type'] = NvInferModelType.CUSTOM.value

    # or configure regular model
    else:
        nvinfer_config['property']['output-tensor-meta'] = 0

        # classifier
        if issubclass(model_type, AttributeModel):
            nvinfer_config['property'][
                'network-type'
            ] = NvInferModelType.CLASSIFIER.value
            # nvinfer doesn't support per attribute threshold
            # workaround: set the overall classifier threshold for regular classifier
            # using the threshold of the first attribute
            for attr in model_config.output.attributes:
                if attr.threshold is not None:
                    nvinfer_config['property']['classifier-threshold'] = attr.threshold
                    break

        # instance segmentation
        elif model_type == NvInferInstanceSegmentation:
            nvinfer_config['property'][
                'network-type'
            ] = NvInferModelType.INSTANCE_SEGMENTATION.value
            # clustering is done by the model itself
            nvinfer_config['property']['cluster-mode'] = 4

        # detector
        else:
            nvinfer_config['property']['network-type'] = NvInferModelType.DETECTOR.value
            # set NMS clustering (so far only this one is supported)
            nvinfer_config['property']['cluster-mode'] = 2

    if module_config.parameters.dev_mode:
        if model_config.input.preprocess_object_meta:
            model_config.input.preprocess_object_meta.dev_mode = True
        if model_config.input.preprocess_object_image:
            model_config.input.preprocess_object_image.dev_mode = True
        if model_config.output.converter:
            model_config.output.converter.dev_mode = True
        if issubclass(model_type, (ObjectModel, ComplexModel)):
            for obj in model_config.output.objects:
                if obj.selector:
                    obj.selector.dev_mode = True

    element_config.model = model_config

    # save resulting nvinfer config file
    # build config file name using required model engine file
    model_name = model_config.engine_file.split('.')[0]
    config_file = f'{model_name}_config_savant.txt'
    config_file_path = Path(model_config.local_path) / config_file
    NvInferConfig.write_file(nvinfer_config, config_file_path)
    logger.info('Resulting configuration file "%s" has been saved.', config_file_path)

    if 'properties' not in element_config:
        element_config.properties = {}
    element_config.properties['config-file-path'] = str(config_file_path)

    return element_config


def nvtracker_element_configurator(
    element_config: DictConfig,
    module_config: DictConfig,
) -> DictConfig:
    ll_config_file = element_config.properties.get('ll-config-file')
    if ll_config_file is not None and not os.path.exists(ll_config_file):
        raise NvTrackerConfigException(
            f'File not found when loading low-level library config for nvtracker: {ll_config_file}.'
            ' Please check the path to the file.'
        )

    ll_lib_file = element_config.properties.get('ll-lib-file')
    if ll_lib_file is not None and not os.path.exists(ll_lib_file):
        raise NvTrackerConfigException(
            f'File not found when loading low-level library for nvtracker: {ll_lib_file}.'
            ' Please check the path to the file.'
        )

    return element_config
