"""Logger utils."""
import logging
import logging.config
import os
from typing import Optional


def _get_default_loglevel() -> str:
    return os.environ.get('LOGLEVEL', 'INFO')


def _log_conf(log_level: str) -> dict:
    return {
        'version': 1,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] '
                '%(message)s',
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
        'disable_existing_loggers': False,
    }


def init_logging(log_level: Optional[str] = None):
    """Initialize logging with specified log level or set default.

    :param log_level: One of supported by logging module: INFO, DEBUG, etc.
    """
    if init_logging.done:
        return

    level = _get_default_loglevel() if log_level is None else log_level.upper()
    log_config = {
        'root': {'level': level, 'handlers': ['console']},
    }
    log_config.update(_log_conf(level))
    logging.config.dictConfig(log_config)
    init_logging.done = True


init_logging.done = False


def update_logging(log_level: str):
    """Update logging with specified log level.

    :param log_level: One of supported by logging module: INFO, DEBUG, etc.
    """
    logging.config.dictConfig(_log_conf(log_level.upper()))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    init_logging()
    return logging.getLogger(name)


class LoggerMixin:
    """Mixes logger in GStreamer element.

    When the element name is available, logger name changes to
    `module_name/element_name`. Otherwise, logger name is `module_name`.

    Note: we cannot override `do_set_state` or any other method where element name
    becomes available since base classes are bindings.
    """

    _logger: logging.Logger = None
    _logger_initialized: bool = False

    def __init__(self):
        self._init_logger()

    @property
    def logger(self):
        """Logger."""
        if not self._logger_initialized:
            self._init_logger()
        return self._logger

    def _init_logger(self):
        logger_name = f'savant.{self.__module__}'
        if hasattr(self, 'get_name') and self.get_name():
            logger_name += f'.{self.get_name()}'
        self._logger = get_logger(logger_name)

        self._logger_initialized = True
