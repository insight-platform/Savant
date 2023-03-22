"""Module entrypoint.

>>> python -m savant.entrypoint [config_file_path]
"""
import logging
import logging.config
import sys
from savant.config import ModuleConfig
from savant.gstreamer import Gst
from savant.gstreamer.runner import GstPipelineRunner
from savant.deepstream.pipeline import NvDsPipeline
from savant.utils.sink_factories import sink_factory


def log_conf(log_level: str):
    return {
        'version': 1,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'detailed',
                'stream': 'ext://sys.stdout',
            },
        },
        'loggers': {
            'savant': {
                'level': log_level,
                'handlers': ['console'],
                'propagate': False,
            },
        },
    }


def main(config_file_path: str):
    """Entrypoint for NvDsPipeline based module.

    :param config_file_path: Module configuration file path.
    """
    # load default.yml and set up logging
    config = ModuleConfig().config

    log_config = {
        'root': {'level': 'INFO', 'handlers': ['console']},
    }
    log_config.update(log_conf(config.parameters['log_level'].upper()))
    logging.config.dictConfig(log_config)

    # load module config
    config = ModuleConfig().load(config_file_path)
    # reconfigure savant logger with updated loglevel
    logging.config.dictConfig(log_conf(config.parameters['log_level'].upper()))
    logger = logging.getLogger('savant')

    # possible exceptions will cause app to crash and log error by default
    # no need to handle exceptions here
    sink = sink_factory(config.pipeline.sink)

    Gst.init(None)

    pipeline = NvDsPipeline(
        name=config.name,
        source=config.pipeline.source,
        elements=config.pipeline.elements,
        **config.parameters,
    )

    try:
        with GstPipelineRunner(pipeline):
            for msg in pipeline.stream():
                sink(msg, **dict(module_name=config.name))
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(exc, exc_info=True)


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        print('Module config file path is expected as a CLI argument.')
