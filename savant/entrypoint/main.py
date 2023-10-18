"""Module entrypoint function."""
import os
import signal
from pathlib import Path
from threading import Thread

from savant.config import ModuleConfig
from savant.deepstream.encoding import check_encoder_is_available
from savant.deepstream.pipeline import NvDsPipeline
from savant.deepstream.runner import NvDsPipelineRunner
from savant.gstreamer import Gst
from savant.healthcheck.server import HealthCheckHttpServer
from savant.healthcheck.status import ModuleStatus, set_module_status
from savant.utils.logging import get_logger, init_logging, update_logging
from savant.utils.sink_factories import sink_factory


def main(config_file_path: str, *args):
    """Entrypoint for NvDsPipeline based module.

    :param config_file_path: Module configuration file path.
    :param args: Config overrides in dot-list format
    """

    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
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
    config = ModuleConfig().load(config_file_path, *args)

    # reconfigure savant logger with updated loglevel
    update_logging(config.parameters['log_level'])
    logger = get_logger('main')

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
