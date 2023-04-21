"""Logger utils."""
from typing import Optional
import logging
import logging.config


DEFAULT_LOGLEVEL = 'INFO'


def _log_conf(log_level: str) -> dict:
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


def init_logging(log_level: Optional[str] = None):
    """Initialize logging with specified log level or set default.

    :param log_level: One of supported by logging module: INFO, DEBUG, etc.
    """
    level = DEFAULT_LOGLEVEL if log_level is None else log_level.upper()
    log_config = {
        'root': {'level': level, 'handlers': ['console']},
    }
    log_config.update(_log_conf(level))
    logging.config.dictConfig(log_config)


def update_logging(log_level: str):
    """Update logging with specified log level.

    :param log_level: One of supported by logging module: INFO, DEBUG, etc.
    """
    logging.config.dictConfig(_log_conf(log_level.upper()))
