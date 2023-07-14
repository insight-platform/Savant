"""Module configuration."""
import re
from pathlib import Path
from typing import Callable, Optional, Union, Tuple, Type
import logging
from omegaconf import OmegaConf, DictConfig
from savant.config.schema import (
    Module,
    SimplePipeline,
    CompositePipeline,
    PipelineElement,
    PyFuncElement,
    ModelElement,
    get_element_name,
    DrawFunc,
    FrameParameters,
)
from savant.deepstream.nvinfer.element_config import nvinfer_configure_element
from savant.parameter_storage import init_param_storage
from savant.utils.singleton import SingletonMeta

logger = logging.getLogger(__name__)


class ModuleConfigException(Exception):
    """Module config exception class."""


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
    logger.debug(
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
            'Parsed full definiton, result element="%s" elem_type="%s" elem_ver="%s"',
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
        return PyFuncElement, None

    if element == 'nvinfer':
        return ModelElement, nvinfer_configure_element

    return PipelineElement, None


def setup_batch_size(config: Module) -> None:
    """Setup/check module batch size. Call this only after calling to_object
    due to the need to post initialize model.

    :param config: module config
    """

    def iterate_elements(elements):
        for element in elements:
            if isinstance(element, ModelElement):
                return element.model.batch_size
        return None

    first_model_batch_size = None
    if isinstance(config.pipeline, SimplePipeline):
        first_model_batch_size = iterate_elements(config.pipeline.elements)
    elif isinstance(config.pipeline, CompositePipeline):
        for stage in config.pipeline.stages:
            first_model_batch_size = iterate_elements(stage.elements)
            if first_model_batch_size is not None:
                break

    parameter_batch_size = config.parameters.get('batch_size')

    if first_model_batch_size is not None:
        if (
            parameter_batch_size is not None
            and parameter_batch_size != first_model_batch_size
        ):
            raise ModuleConfigException(
                'Module parameter "batch_size" is set explicitly '
                'and does not match first pipeline model "batch_size".'
            )
        batch_size = first_model_batch_size
    elif parameter_batch_size is None:
        raise ModuleConfigException('Parameter "batch_size" is required.')
    else:
        batch_size = parameter_batch_size

    if not (0 < batch_size <= 1024):
        raise ModuleConfigException(
            f'Value "{batch_size}" is out of range '
            'for parameter "batch_size". Allowed values: 1 - 1024.'
        )
    config.parameters['batch_size'] = batch_size


def resolve_parameters(config: DictConfig):
    """Resolve parameters on module config ("frame", "draw_func", etc.).

    :param config: Module config.
    """

    config.parameters['frame'] = OmegaConf.unsafe_merge(
        OmegaConf.structured(FrameParameters),
        config.parameters['frame'],
    )

    draw_func_cfg = config.parameters.get('draw_func')
    if draw_func_cfg is not None:
        draw_func_schema = OmegaConf.structured(DrawFunc)
        draw_func_cfg = OmegaConf.unsafe_merge(draw_func_schema, draw_func_cfg)
        config.parameters['draw_func'] = draw_func_cfg


def elements_type_config(elements):
    """Convert elements to proper type"""

    configured_elements = []
    for element_config in elements:
        try:
            element, elem_type, elem_ver = get_elem_type_ver(element_config)

            element_config.element = element
            if elem_type:
                element_config.element_type = elem_type
            if elem_ver:
                element_config.version = elem_ver

            element_schema, configurator = get_schema_configurator(element)

            element_config = OmegaConf.unsafe_merge(element_schema, element_config)

            if element_config.version != 'v1':
                raise ModuleConfigException('Only version "v1" is supported.')

            if configurator:
                element_config = configurator(element_config)

            configured_elements.append(element_config)

        except Exception as exc:
            raise ModuleConfigException(
                f'Element {get_element_name(element_config)}: {exc}'
            ) from exc

    return configured_elements


class ModuleConfig(metaclass=SingletonMeta):
    """Singleton that provides module configuration loading and access."""

    def __init__(self):
        self._config_schema = OmegaConf.structured(Module)
        self._default_cfg = OmegaConf.load(
            Path(__file__).parent.resolve() / 'default.yml'
        )
        logger.debug('loaded default config\n%s', OmegaConf.to_yaml(self._default_cfg))
        self._config = None

    def load(self, config_file_path: Union[str, Path]) -> Module:
        """Loads and prepares module configuration.

        :param config_file_path: Module config file path
        :return: Module configuration, structured
        """
        module_cfg = OmegaConf.load(config_file_path)
        logger.debug('loaded module config\n%s', OmegaConf.to_yaml(module_cfg))

        # if source for module is specified,
        # it should be used instead of default source (not merged)
        if 'pipeline' in module_cfg:
            if module_cfg.pipeline is None:
                module_cfg.pipeline = {}
            if 'source' in module_cfg.pipeline:
                del self._default_cfg.pipeline.source

        if 'pipeline' in module_cfg:
            if 'stages' in module_cfg.pipeline and 'elements' in module_cfg.pipeline:
                raise ModuleConfigException(
                    '"stages" and "elements" pipeline config keys are mutually exclusive.'
                )

            if 'stages' in module_cfg.pipeline:
                pipeline_schema = OmegaConf.structured(CompositePipeline)
            else:
                pipeline_schema = OmegaConf.structured(SimplePipeline)

            pipeline_cfg = OmegaConf.merge(
                pipeline_schema, self._default_cfg.pipeline, module_cfg.pipeline
            )
            del module_cfg.pipeline
        else:
            pipeline_cfg = OmegaConf.merge(
                OmegaConf.structured(SimplePipeline), self._default_cfg.pipeline
            )
        del self._default_cfg.pipeline

        logger.info('Configure module...')

        module_cfg = OmegaConf.unsafe_merge(
            self._config_schema, self._default_cfg, module_cfg
        )
        module_cfg.pipeline = pipeline_cfg

        init_param_storage(module_cfg)

        OmegaConf.resolve(module_cfg)  # to resolve parameters for pipeline elements
        resolve_parameters(module_cfg)

        logger.info('Configure pipeline elements...')

        if (
            OmegaConf.get_type(module_cfg.pipeline).__name__
            == CompositePipeline.__name__
        ):
            for stage in module_cfg.pipeline.stages:
                stage.elements = elements_type_config(stage.elements)
        else:
            module_cfg.pipeline.elements = elements_type_config(
                module_cfg.pipeline.elements
            )

        self._config = OmegaConf.to_object(module_cfg)

        setup_batch_size(self._config)

        logger.info('Module configuration is complete.')
        logger.debug('Module config:\n%s', OmegaConf.to_yaml(self._config))

        return self._config

    @property
    def config(self):
        """Before load returns default config.

        After load returns loaded config.
        """
        if self._config is not None:
            return self._config
        return self._default_cfg
