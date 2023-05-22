"""Module configuration."""
import re
from pathlib import Path
from typing import Callable, Optional, Union, Tuple, Type
import logging
from omegaconf import OmegaConf, DictConfig
from savant.config.schema import (
    Module,
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
    first_model_batch_size = None
    for element in config.pipeline.elements:
        if isinstance(element, ModelElement) and first_model_batch_size is None:
            first_model_batch_size = element.model.batch_size
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


class ModuleConfig(metaclass=SingletonMeta):
    """Singleton that provides module configuration loading and access."""

    def __init__(self):
        self._config_schema = OmegaConf.structured(Module)
        self._default_cfg = OmegaConf.load(
            Path(__file__).parent.resolve() / 'default.yml'
        )
        self._config = None

    def load(self, config_file_path: Union[str, Path]) -> Module:
        """Loads and prepares module configuration.

        :param config_file_path: Module config file path
        :return: Module configuration, structured
        """
        module_cfg = OmegaConf.load(config_file_path)

        logger.info('Configure module...')

        # if source for module is specified,
        # it should be used instead of default source (not merged)
        if 'pipeline' in module_cfg and 'source' in module_cfg.pipeline:
            del self._default_cfg.pipeline.source

        logger.debug('Default conf\n%s', self._default_cfg)
        logger.debug('Module conf\n%s', module_cfg)
        module_cfg = OmegaConf.unsafe_merge(
            self._config_schema, self._default_cfg, module_cfg
        )
        logger.debug('Merged conf\n%s', module_cfg)
        init_param_storage(module_cfg)

        OmegaConf.resolve(module_cfg)  # to resolve parameters for pipeline elements
        resolve_parameters(module_cfg)
        logger.debug('Resolved conf\n%s', module_cfg)
        logger.info('Configure pipeline elements...')

        # convert elements to proper type
        for i, element_config in enumerate(module_cfg.pipeline.elements):
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

                module_cfg.pipeline.elements[i] = element_config

            except Exception as exc:
                raise ModuleConfigException(
                    f'Element {get_element_name(element_config)}: {exc}'
                ) from exc

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
