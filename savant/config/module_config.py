"""Module configuration."""

import re
from pathlib import Path
from typing import IO, Any, Callable, Iterable, Optional, Tuple, Type, Union

from omegaconf import DictConfig, OmegaConf

from savant.config.schema import (
    BufferQueuesParameters,
    DrawFunc,
    ElementGroup,
    FrameParameters,
    ModelElement,
    Module,
    Pipeline,
    PipelineElement,
    PyFuncElement,
    SinkElement,
    SourceElement,
    TelemetryParameters,
    get_element_name,
)
from savant.deepstream.nvinfer.element_config import (
    nvinfer_element_configurator,
    nvtracker_element_configurator,
)
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec
from savant.parameter_storage import init_param_storage
from savant.utils.logging import get_logger
from savant.utils.singleton import SingletonMeta
from savant.utils.sink_factories import SINK_REGISTRY

logger = get_logger(__name__)


class ModuleConfigException(Exception):
    """Module config exception class."""


def pyfunc_element_configurator(
    element_config: DictConfig, module_config: DictConfig
) -> DictConfig:
    """Additional configuration steps for PyfuncElements."""
    # if dev mode is enabled in the module parameters
    # set dev mode for the pyfunc element
    if module_config.parameters.dev_mode:
        logger.debug(
            'Setting dev mode for PyFuncElement named "%s" to True.',
            element_config.name,
        )
        element_config.dev_mode = True
    return element_config


def source_element_configurator(
    element_config: DictConfig, module_config: DictConfig
) -> DictConfig:
    """Additional configuration steps for SourceElements."""
    # if dev mode is enabled in the module parameters
    # set dev mode for the ingress filter function
    if (
        element_config.ingress_frame_filter is not None
        and module_config.parameters.dev_mode
    ):
        logger.debug(
            'Setting dev mode for ingress filter of SourceElement named "%s" to True.',
            element_config.name,
        )
        element_config.ingress_frame_filter.dev_mode = True
    return element_config


def sink_element_configurator(
    element_config: DictConfig, module_config: DictConfig
) -> DictConfig:
    """Additional configuration steps for SinkElements."""
    # if dev mode is enabled in the module parameters
    # set dev mode for the egress filter function
    if module_config.parameters.dev_mode:
        logger.debug(
            'Setting dev mode for egress filter of SinkElement named "%s" to True.',
            element_config.name,
        )
        element_config.egress_frame_filter.dev_mode = True
    return element_config


def parse_element_short_notation(
    short_notation: str,
    re_pattern: str = r'(?P<element>\w+)@?(?P<type>\w+)?:?(?P<version>\w+)?',
) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse short notation for element + element_type + element_version.

    Examples:
    ``nvinfer`` -> ``nvinfer, None, None``
    ``nvinfer@attribute_model`` -> ``nvinfer, attribute_model, None``
    ``nvinfer@attribute_model:v1`` -> ``nvinfer, attribute_model, v1``
    ``drawbin:v1`` -> ``drawbin, None, v1``

    :param short_notation: string ``element<@element_type><:element_version>``
    :param re_pattern: regex pattern to parse the short notation string
    :return: parsed element + element_type + element_version
    """
    try:
        elem, elem_type, elem_ver = re.match(
            re_pattern, short_notation, re.ASCII
        ).groups()
    except AttributeError as exc:
        raise ModuleConfigException(
            f'Short notation "{short_notation}" parse failed.'
        ) from exc

    logger.debug(
        'Parsed short notation %s, result element="%s" elem_type="%s" elem_ver="%s"',
        short_notation,
        elem,
        elem_type,
        elem_ver,
    )
    return elem, elem_type, elem_ver


def get_elem_type_ver(
    element_config: DictConfig,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Get element + element_type + element_version from element config.

    :param element_config: element config
    :return: element + element_type + element_version
    """
    logger.trace(
        'Getting element/elem_type/elem_ver from element config %s', element_config
    )

    element = element_config.element.lower()

    elem_type = None
    if 'element_type' in element_config and element_config.element_type:
        elem_type = element_config.element_type.lower()

    elem_ver = None
    if 'version' in element_config and element_config.version:
        elem_ver = element_config.version.lower()

    # to prevent mixing of short notation and full definition
    # only try to parse element if both type and version are not set
    if elem_type or elem_ver:
        if '@' in element or ':' in element:
            raise ModuleConfigException(
                'Mixing short notation and full definition is not supported.'
            )
        logger.debug(
            'Parsed full definition, result element="%s" elem_type="%s" elem_ver="%s"',
            element,
            elem_type,
            elem_ver,
        )
        return element, elem_type, elem_ver

    return parse_element_short_notation(element)


def get_schema_configurator(
    element: str,
) -> Tuple[Type[PipelineElement], Optional[Callable]]:
    """Get element schema and configurator from element.

    :param element: gst element name (gst factory name)
    :return: schema + optional configurator callable
    """

    if element == 'pyfunc':
        return PyFuncElement, pyfunc_element_configurator

    if element == 'nvinfer':
        return ModelElement, nvinfer_element_configurator

    if element == 'nvtracker':
        return PipelineElement, nvtracker_element_configurator

    if element == 'zeromq_source_bin':
        return SourceElement, source_element_configurator

    if element in SINK_REGISTRY:
        return SinkElement, sink_element_configurator

    return PipelineElement, None


def setup_batch_size(config: Module) -> None:
    """Setup/check module batch size. Call this only after calling to_object
    due to the need to post initialize model.

    :param config: module config.
    """

    def find_first_model_element(pipeline: Pipeline) -> Optional[ModelElement]:
        for item in pipeline.elements:
            if isinstance(item, ModelElement):
                return item
            elif isinstance(item, ElementGroup):
                if item.init_condition.is_enabled:
                    for element in item.elements:
                        if isinstance(element, ModelElement):
                            return element
        return None

    batch_size = config.parameters.get('batch_size')
    logger.debug('Pipeline batch size parameter is %s.', batch_size)

    if batch_size is None:
        first_model_element = find_first_model_element(config.pipeline)
        if first_model_element is not None:
            batch_size = first_model_element.model.batch_size
            logger.debug(
                'Found first ModelElement "%s" of the pipeline with '
                'the batch size of %s.',
                first_model_element.name,
                batch_size,
            )
        else:
            logger.debug('No ModelElements found in the pipeline.')

    if batch_size is None:
        raise ModuleConfigException('Parameter "batch_size" is required.')

    if not (0 < batch_size <= 1024):
        raise ModuleConfigException(
            f'Value "{batch_size}" is out of range '
            'for parameter "batch_size". Allowed values: 1 - 1024.'
        )
    config.parameters['batch_size'] = batch_size
    logger.info('Pipeline batch size is set to %s.', batch_size)


def configure_module_parameters(module_cfg: DictConfig) -> None:
    """Resolve parameters on module config ("frame", "draw_func", etc.).

    :param module_cfg: Module config.
    """
    if 'parameters' not in module_cfg or module_cfg.parameters is None:
        module_cfg.parameters = {}
        return

    def apply_schema(
        cfg: dict, node: str, schema_class: Any, default: Any = None
    ) -> None:
        if node not in cfg or cfg[node] is None:
            cfg[node] = default
        else:
            cfg[node] = OmegaConf.unsafe_merge(
                OmegaConf.structured(schema_class),
                cfg[node],
            )

    apply_schema(
        module_cfg.parameters,
        'frame',
        FrameParameters,
        OmegaConf.structured(FrameParameters),
    )
    output_frame = module_cfg.parameters.get('output_frame')
    is_pass_through_mode = output_frame and output_frame['codec'] == 'copy'
    frame_parameters: FrameParameters = module_cfg.parameters['frame']
    if frame_parameters.padding:
        if is_pass_through_mode:
            if frame_parameters.padding.keep:
                logger.warning('Padding keep is ignored in pass-through mode.')
            frame_parameters.padding.keep = False

    apply_schema(module_cfg.parameters, 'draw_func', DrawFunc)
    if module_cfg.parameters.dev_mode and module_cfg.parameters.draw_func:
        logger.debug('Setting draw_func dev mode to true.')
        module_cfg.parameters.draw_func.dev_mode = True
        if is_pass_through_mode:
            logger.warning(
                'The pipeline is configured in video pass-through mode. '
                'In this mode, frame modifications exist only in the '
                'pipeline but are not propagated through the sinks.'
            )
    apply_schema(module_cfg.parameters, 'buffer_queues', BufferQueuesParameters)
    apply_schema(
        module_cfg.parameters,
        'telemetry',
        TelemetryParameters,
        OmegaConf.structured(TelemetryParameters),
    )


def configure_element(
    element_config: DictConfig, module_config: DictConfig
) -> DictConfig:
    """Convert element to proper type.

    :param element_config: element config as read from yaml.
    :param module_config: full module config in case context is required.
    :return: finished element config.
    """
    try:
        element, elem_type, elem_ver = get_elem_type_ver(element_config)

        element_config.element = element
        if elem_type:
            element_config.element_type = elem_type
        if elem_ver:
            element_config.version = elem_ver

        logger.info('Getting schema/configurator for %s', element)
        element_schema, configurator = get_schema_configurator(element)

        element_config = OmegaConf.unsafe_merge(element_schema, element_config)
        OmegaConf.resolve(element_config)

        if element_config.version != 'v1':
            raise ModuleConfigException('Only version "v1" is supported.')

        if configurator:
            element_config = configurator(element_config, module_config)

        return element_config

    except Exception as exc:
        raise ModuleConfigException(
            f'Element {get_element_name(element_config)}: {exc}'
        ) from exc


def merge_configs(
    user_cfg_parts: Iterable[DictConfig], default_cfg: DictConfig
) -> DictConfig:
    """Merge user config parts with default config.

    :param user_cfg_parts: user config parts.
    :param default_cfg: default config.
    :return: merged config.
    """
    user_cfg = OmegaConf.unsafe_merge(*user_cfg_parts)

    # if source for module is specified,
    # it should be used instead of default source (not merged)
    if 'pipeline' in user_cfg and 'source' in user_cfg.pipeline:
        if user_cfg.pipeline.source:
            del default_cfg.pipeline.source
        else:
            del user_cfg.pipeline.source

    return OmegaConf.unsafe_merge(default_cfg, user_cfg)


def configure_pipeline(module_cfg: DictConfig) -> None:
    """Convert pipeline elements to proper types.

    :param module_cfg: module config
    """
    if 'pipeline' not in module_cfg or module_cfg.pipeline is None:
        module_cfg.pipeline = OmegaConf.structured(Pipeline)
        return

    logger.debug('Configuring pipeline source...')
    module_cfg.pipeline.source = configure_element(
        module_cfg.pipeline.source, module_cfg
    )

    logger.debug('Configuring pipeline sink...')
    for pipeline_sink_idx, item in enumerate(module_cfg.pipeline.sink):
        item_cfg = configure_element(item, module_cfg)
        module_cfg.pipeline.sink[pipeline_sink_idx] = item_cfg

    logger.debug('Configuring pipeline elements...')
    if 'elements' not in module_cfg.pipeline or module_cfg.pipeline.elements is None:
        module_cfg.pipeline.elements = []
        return
    group_schema = OmegaConf.structured(ElementGroup)
    for pipeline_el_idx, item in enumerate(module_cfg.pipeline.elements):
        if 'element' in item:
            item_cfg = configure_element(item, module_cfg)
        elif 'group' in item:
            if 'elements' in item.group or item.group.elements is not None:
                for grp_element_idx, grp_element in enumerate(item.group.elements):
                    element_cfg = configure_element(grp_element, module_cfg)
                    item.group.elements[grp_element_idx] = element_cfg
            else:
                item.group.elements = []
            item_cfg = OmegaConf.merge(group_schema, item.group)
        else:
            raise ModuleConfigException(
                f'Config node under "pipeline.elements" should include either'
                f' "element" or "group". Config node: {item}.'
            )
        module_cfg.pipeline.elements[pipeline_el_idx] = item_cfg


def validate_frame_parameters(config: Module):
    """Validate frame parameters."""

    frame_parameters: FrameParameters = config.parameters['frame']
    if frame_parameters.width % frame_parameters.geometry_base != 0:
        raise ModuleConfigException(
            f'Frame width ({frame_parameters.width}) '
            f'must be divisible by geometry base ({frame_parameters.geometry_base}).'
        )
    if frame_parameters.height % frame_parameters.geometry_base != 0:
        raise ModuleConfigException(
            f'Frame height ({frame_parameters.height}) '
            f'must be divisible by geometry base ({frame_parameters.geometry_base}).'
        )
    if frame_parameters.padding is None:
        return

    if frame_parameters.padding.left % frame_parameters.geometry_base != 0:
        raise ModuleConfigException(
            f'Left padding ({frame_parameters.padding.left}) '
            f'must be divisible by geometry base ({frame_parameters.geometry_base}).'
        )
    if frame_parameters.padding.right % frame_parameters.geometry_base != 0:
        raise ModuleConfigException(
            f'Right padding ({frame_parameters.padding.right}) '
            f'must be divisible by geometry base ({frame_parameters.geometry_base}).'
        )
    if frame_parameters.padding.top % frame_parameters.geometry_base != 0:
        raise ModuleConfigException(
            f'Top padding ({frame_parameters.padding.top}) '
            f'must be divisible by geometry base ({frame_parameters.geometry_base}).'
        )
    if frame_parameters.padding.bottom % frame_parameters.geometry_base != 0:
        raise ModuleConfigException(
            f'Bottom padding ({frame_parameters.padding.bottom}) '
            f'must be divisible by geometry base ({frame_parameters.geometry_base}).'
        )


def validate_output_frame_parameters(config: Module):
    output_frame = config.parameters.get('output_frame')
    if not output_frame:
        return

    codec = output_frame['codec']
    if codec != 'copy' and codec not in CODEC_BY_NAME:
        raise ModuleConfigException(f'Unknown codec {codec!r} in output_frame config.')

    profile = output_frame.get('profile')
    if profile is not None:
        if codec != Codec.H264.value.name:
            raise ModuleConfigException(
                f'Profile can be configured only for {Codec.H264.value.name!r} codec.'
            )
        supported_profiles = ['baseline', 'main', 'high']
        if profile not in supported_profiles:
            raise ModuleConfigException(
                f'Profile {profile!r} is not supported. '
                f'Supported profiles: {supported_profiles}.'
            )


class ModuleConfig(metaclass=SingletonMeta):
    """Singleton that provides module configuration loading and access."""

    def __init__(self):
        self._default_cfg = OmegaConf.load(
            Path(__file__).parent.resolve() / 'default.yml'
        )
        logger.debug('Loaded default config\n%s', OmegaConf.to_yaml(self._default_cfg))
        self._config = None

    def load(self, config: Union[str, Path, IO[Any]]) -> Module:
        """Loads and prepares module configuration.

        :param config: Module config
        :return: Module configuration, structured
        """
        module_cfg = OmegaConf.load(config)
        logger.info('Configuring module "%s"...', module_cfg.name)
        module_cfg = merge_configs([module_cfg], self._default_cfg)

        module_cfg = OmegaConf.unsafe_merge(OmegaConf.structured(Module), module_cfg)

        init_param_storage(module_cfg)
        OmegaConf.resolve(module_cfg)  # to resolve parameters for pipeline elements

        logger.debug('Configuring module parameters...')
        configure_module_parameters(module_cfg)

        logger.debug('Configuring pipeline...')
        configure_pipeline(module_cfg)

        self._config = OmegaConf.to_object(module_cfg)

        validate_frame_parameters(self._config)
        validate_output_frame_parameters(self._config)

        setup_batch_size(self._config)

        logger.info('Module configuration is complete.')
        logger.debug('Module config:\n%s', OmegaConf.to_yaml(self._config))

        return self._config

    @property
    def config(self) -> Union[Module, DictConfig]:
        """Before load returns default config.

        After load returns loaded config.
        """
        if self._config is not None:
            return self._config
        return self._default_cfg
