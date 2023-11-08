"""Module entrypoints."""
import os
import signal
from pathlib import Path
from threading import Thread
from typing import IO, Any, Union

from savant.config import ModuleConfig
from savant.config.schema import ElementGroup, ModelElement
from savant.deepstream.encoding import check_encoder_is_available
from savant.deepstream.nvinfer.build_engine import build_engine
from savant.deepstream.nvinfer.model import NvInferModel
from savant.deepstream.pipeline import NvDsPipeline
from savant.deepstream.runner import NvDsPipelineRunner
from savant.gstreamer import Gst
from savant.healthcheck.server import HealthCheckHttpServer
from savant.healthcheck.status import ModuleStatus, set_module_status
from savant.utils.check_display import check_display_env
from savant.utils.logging import get_logger, init_logging, update_logging
from savant.utils.sink_factories import sink_factory


def run_module(module_config: Union[str, Path, IO[Any]]):
    """Runs NvDsPipeline based module.

    :param module_config: Module configuration.
    """

    # To gracefully shut down the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    status_filepath = os.environ.get('SAVANT_STATUS_FILEPATH')
    if status_filepath is not None:
        status_filepath = Path(status_filepath)
        if status_filepath.exists():
            status_filepath.unlink()
        status_filepath.parent.mkdir(parents=True, exist_ok=True)
        set_module_status(status_filepath, ModuleStatus.INITIALIZING)

    # load default.yml and set up logging
    config = ModuleConfig().config

    init_logging(config.parameters['log_level'])

    # load module config
    config = ModuleConfig().load(module_config)

    # reconfigure savant logger with updated loglevel
    update_logging(config.parameters['log_level'])
    logger = get_logger('main')

    check_display_env(logger)

    if status_filepath is not None:
        healthcheck_port = config.parameters.get('healthcheck_port')
        if healthcheck_port:
            healthcheck_server = HealthCheckHttpServer(
                host='',
                port=healthcheck_port,
                http_path='/healthcheck',
                status_filepath=status_filepath,
            )
            healthcheck_thread = Thread(
                target=healthcheck_server.serve_forever,
                daemon=True,
            )
            healthcheck_thread.start()

    # possible exceptions will cause app to crash and log error by default
    # no need to handle exceptions here
    sink = sink_factory(config.pipeline.sink)

    Gst.init(None)

    if not check_encoder_is_available(config.parameters):
        return

    pipeline = NvDsPipeline(
        config.name,
        config.pipeline,
        **config.parameters,
    )

    try:
        with NvDsPipelineRunner(pipeline, status_filepath) as runner:
            try:
                for msg in pipeline.stream():
                    sink(msg, **dict(module_name=config.name))
            except KeyboardInterrupt:
                logger.info('Shutting down.')
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(exc, exc_info=True)
                # TODO: Sometimes pipeline hangs when exit(1) or not exit at all is called.
                #       E.g. when the module has "req+connect" socket at the sink and
                #       sink adapter is not available.
                os._exit(1)  # pylint: disable=protected-access
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(exc, exc_info=True)
        exit(1)

    if runner.error is not None:
        exit(1)


def build_module_engines(module_config: Union[str, Path, IO[Any]]):
    """Builds module model's engines.

    :param module_config: Module configuration.
    """

    # To gracefully shut down the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    # load default.yml and set up logging
    config = ModuleConfig().config

    init_logging(config.parameters['log_level'])

    # load module config
    config = ModuleConfig().load(module_config)

    # reconfigure savant logger with updated loglevel
    update_logging(config.parameters['log_level'])
    logger = get_logger('engine_builder')

    check_display_env(logger)

    nvinfer_elements = []
    for element in config.pipeline.elements:
        if isinstance(element, ModelElement):
            if isinstance(element.model, NvInferModel):
                nvinfer_elements.append(element)

        elif isinstance(element, ElementGroup):
            if element.init_condition.is_enabled:
                for group_element in element.elements:
                    if isinstance(group_element, ModelElement):
                        if isinstance(group_element.model, NvInferModel):
                            nvinfer_elements.append(group_element)

    if not nvinfer_elements:
        logger.error('No model elements found.')
        exit(1)

    Gst.init(None)
    logger.info('GStreamer initialization done.')

    for element in nvinfer_elements:
        logger.info('Start building of the "%s" model engine.', element.name)
        try:
            build_engine(element)
            logger.info(
                'Successfully complete the engine building of the "%s" model.',
                element.name,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                'Failed to build the model "%s" engine: %s',
                element.name,
                exc,
                exc_info=True,
            )
            exit(1)
