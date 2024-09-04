"""Gst-nvinfer file configuration."""

import copy
import re
from configparser import ConfigParser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableSequence, Optional, Union

from omegaconf import DictConfig

from savant.base.model import ModelColorFormat, ModelPrecision

from .model import NVINFER_DEFAULT_OBJECT_SELECTOR, NvInferModel, NvInferModelFormat

__all__ = ['NvInferConfig', 'NvInferConfigType']

# nvinfer config type specification
NvInferConfigType = Dict[str, Dict]


class NvInferConfig:
    """DeepStream Gst-nvinfer File Configuration manager.

    https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html
    """

    @staticmethod
    def default() -> NvInferConfigType:
        """Returns default config."""
        return {'property': {}}

    @staticmethod
    def read_file(config_file_path: Union[Path, str]) -> NvInferConfigType:
        """Reads and parses nvinfer model config file.

        :param config_file_path: path to config file
        :return: dict {section: [sub-section:] {property : value}}
        """

        config_parser = ConfigParser()
        config_parser.read(config_file_path)
        # using config_parser directly is inconvenient due to its limitations,
        # e.g. `option values must be strings`
        return {section: dict(config_parser[section]) for section in config_parser}

    @staticmethod
    def write_file(config: NvInferConfigType, config_file_path: Union[Path, str]):
        """Prepares and writes nvinfer config file."""
        cfg = copy.deepcopy(config)

        def _to_str(dic):
            for key, val in dic.items():
                if isinstance(val, MutableSequence):
                    dic[key] = ';'.join(map(str, val))
                elif isinstance(val, Mapping):
                    dic[key] = _to_str(val)
                elif isinstance(val, bool):
                    dic[key] = str(int(val))
                else:
                    dic[key] = str(val)
            return dic

        config_parser = ConfigParser()
        config_parser.read_dict(_to_str(cfg))

        with open(config_file_path, 'w', encoding='utf8') as configfile:
            config_parser.write(configfile)

    @staticmethod
    def merge(
        config1: NvInferConfigType, config2: NvInferConfigType
    ) -> NvInferConfigType:
        """Merges config2 into config1.

        :param config1: config to merge into
        :param config2: config to merge
        :return: merged configuration
        """

        def deep_update(dic, upd_dic):
            for key, val in upd_dic.items():
                if isinstance(val, Mapping):
                    dic[key] = deep_update(dic.get(key, {}), val)
                elif val is not None:
                    dic[key] = val
            return dic

        return deep_update(config1, config2)

    @staticmethod
    def parse_model_engine_file(model_engine_file: str) -> Optional[Dict]:
        """Try to parse ``model-engine-file`` according to default Deepstream
        engine naming scheme."""
        result = re.match(
            r'(?P<model_file>.*)'
            r'_b(?P<batch_size>\d+)'
            r'_gpu(?P<gpu_id>\d+)'
            r'_(?P<precision>fp32|fp16|int8)\.engine',
            model_engine_file,
        )
        if not result:
            return None
        return {
            'model_file': result['model_file'],
            'batch_size': int(result['batch_size']),
            'gpu_id': int(result['gpu_id']),
            'precision': ModelPrecision[result['precision'].upper()],
        }

    @staticmethod
    def generate_model_engine_file(
        model_file: str, batch_size: int, device_id: str, precision: ModelPrecision
    ) -> str:
        """Generate ``model-engine-file`` according to default Deepstream
        engine naming scheme."""
        prefix = model_file if model_file else 'model'
        return '{}_b{}_{}_{}.engine'.format(
            prefix, batch_size, device_id, precision.name.lower()
        )

    @dataclass
    class _FieldMap:
        """Mapping Gst-nvinfer configuration file properties to model config
        fields."""

        property_name: str
        """Gst-nvinfer configuration file property name ([property] section)."""

        name: str
        """Nvinfer model config field name."""

        converter: Optional[Callable] = None
        """Function to convert property to field."""

        default_value: Optional[Any] = None
        """Default value for missing property."""

    _PROPERTY_FIELD_MAP: List[_FieldMap] = [
        _FieldMap('model-file', 'model_file'),
        _FieldMap('proto-file', 'proto_file'),
        _FieldMap('uff-file', 'model_file'),
        _FieldMap('onnx-file', 'model_file'),
        _FieldMap('tlt-encoded-model', 'model_file'),
        _FieldMap('tlt-model-key', 'tlt_model_key'),
        _FieldMap('custom-network-config', 'custom_config_file'),
        _FieldMap('int8-calib-file', 'int8_calib_file'),
        _FieldMap('model-engine-file', 'engine_file'),
        _FieldMap('mean-file', 'mean_file'),
        _FieldMap('labelfile-path', 'label_file'),
        _FieldMap('batch-size', 'batch_size', int, 1),
        _FieldMap('network-mode', 'precision', lambda v: ModelPrecision(int(v)), 0),
        _FieldMap('custom-lib-path', 'custom_lib_path'),
        _FieldMap('engine-create-func-name', 'engine_create_func_name'),
        _FieldMap('workspace-size', 'workspace_size'),
        _FieldMap('enable-dla', 'enable_dla', lambda v: int(v), 0),
        _FieldMap('use-dla-core', 'use_dla_core', int, 0),
        _FieldMap('scaling-compute-hw', 'scaling_compute_hw', int, 0),
        _FieldMap('scaling-filter', 'scaling_filter', int, 0),
        _FieldMap(
            'parse-bbox-instance-mask-func-name', 'parse_bbox_instance_mask_func_name'
        ),
        _FieldMap('parse-bbox-func-name', 'parse_bbox_func_name'),
        _FieldMap('parse-classifier-func-name', 'parse_classifier_func_name'),
        _FieldMap('uff-input-blob-name', 'input.layer_name'),
        _FieldMap(
            'infer-dims', 'input.shape', lambda v: [int(e) for e in v.split(';')]
        ),
        _FieldMap(
            'maintain-aspect-ratio',
            'input.maintain_aspect_ratio',
            lambda v: bool(int(v)),
            0,
        ),
        _FieldMap(
            'symmetric-padding',
            'input.symmetric_padding',
            lambda v: bool(int(v)),
            0,
        ),
        _FieldMap('net-scale-factor', 'input.scale_factor', float, 1.0),
        _FieldMap(
            'offsets',
            'input.offsets',
            lambda v: [float(e) for e in v.split(';')],
            '0.0;0.0;0.0',
        ),
        _FieldMap(
            'model-color-format',
            'input.color_format',
            lambda v: ModelColorFormat(int(v)),
            0,
        ),
        _FieldMap('input-object-min-width', 'input.object_min_width', int),
        _FieldMap('input-object-min-height', 'input.object_min_height', int),
        _FieldMap('input-object-max-width', 'input.object_max_width', int),
        _FieldMap('input-object-max-height', 'input.object_max_height', int),
        _FieldMap('output-blob-names', 'output.layer_names', lambda v: v.split(';')),
        _FieldMap('num-detected-classes', 'output.num_detected_classes', int),
        _FieldMap('gpu-id', 'gpu_id', int),
        _FieldMap('secondary-reinfer-interval', 'interval', int),
        _FieldMap(
            'layer-device-precision', 'layer_device_precision', lambda v: v.split(';')
        ),
    ]

    _CLASS_ATTR_MAP = [
        _FieldMap('pre-cluster-threshold', 'confidence_threshold', float, 0.5),
        _FieldMap('nms-iou-threshold', 'nms_iou_threshold', float, 0.5),
        _FieldMap('detected-min-w', 'min_width', int, 32),
        _FieldMap('detected-min-h', 'min_height', int, 32),
        _FieldMap('detected-max-w', 'max_width', int),
        _FieldMap('detected-max-h', 'max_height', int),
        _FieldMap('topk', 'top_k', int),
    ]

    @staticmethod
    def to_model(
        nvinfer_config: NvInferConfigType, model_config: NvInferModel
    ) -> NvInferModel:
        """Convert nvinfer config from file format to schema format."""
        config = copy.deepcopy(model_config)

        for field in NvInferConfig._PROPERTY_FIELD_MAP:
            value = nvinfer_config['property'].get(
                field.property_name, field.default_value
            )
            if value is None:
                continue
            if field.converter:
                value = field.converter(value)
            obj = config
            field_names = field.name.split('.')
            for field_name in field_names[:-1]:
                obj = getattr(obj, field_name)
            setattr(obj, field_names[-1], value)

        return config

    @staticmethod
    def from_model(
        model_config: DictConfig, nvinfer_config: Optional[NvInferConfigType] = None
    ) -> NvInferConfigType:
        """Convert nvinfer config from schema format to file format."""
        config = (
            copy.deepcopy(nvinfer_config) if nvinfer_config else NvInferConfig.default()
        )

        for field in NvInferConfig._PROPERTY_FIELD_MAP:
            # filter model file by model format
            if (
                field.property_name == 'uff-file'
                and model_config.format != NvInferModelFormat.UFF
            ):
                continue
            if (
                field.property_name == 'onnx-file'
                and model_config.format != NvInferModelFormat.ONNX
            ):
                continue
            if (
                field.property_name == 'tlt-encoded-model'
                and model_config.format != NvInferModelFormat.ETLT
            ):
                continue
            if field.property_name == 'model-file' and model_config.format not in (
                NvInferModelFormat.CAFFE,
                NvInferModelFormat.CUSTOM,
            ):
                continue

            value = model_config
            field_names = field.name.split('.')
            for field_name in field_names:
                if not value:
                    break
                value = getattr(value, field_name, None)
            if isinstance(value, Enum):
                value = value.value
            if value is not None:
                config['property'][field.property_name] = value

        if 'objects' not in model_config.output or not model_config.output.objects:
            return config

        # setup class-attrs for object model (detector)
        # set a high confidence threshold initially for all classes
        # to filter out only the desired classes
        config['class-attrs-all'] = {'pre-cluster-threshold': 1e10}
        # replace class-attrs parameters with selector kwargs
        for obj in model_config.output.objects:
            class_attrs = {}
            if not obj.selector or 'kwargs' not in obj.selector:
                obj.selector = NVINFER_DEFAULT_OBJECT_SELECTOR
            for field in NvInferConfig._CLASS_ATTR_MAP:
                value = obj.selector.kwargs.get(field.name)
                if value is not None:
                    class_attrs[field.property_name] = value
            if class_attrs:
                config[f'class-attrs-{obj.class_id}'] = class_attrs

        return config
