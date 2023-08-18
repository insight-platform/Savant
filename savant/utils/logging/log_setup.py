import os
from typing import Optional
import logging
from savant_rs.logging import LogLevel, set_log_level
from .log_utils import add_logging_level, log_level_py_to_rs


def get_default_loglevel() -> str:
    return os.environ.get('LOGLEVEL', 'INFO')


def get_log_conf(log_level: str) -> dict:
    return {
        'version': 1,
        'formatters': {
            'basic': {},
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] '
                '%(message)s',
            },
            'pretty_traceback': {
                'class': 'pretty_traceback.LoggingFormatter'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'pretty_traceback',
                'stream': 'ext://sys.stdout',
            },
            'savantrs': {
                'class': 'savant.utils.logging.savant_rs_handler.SavantRsLoggingHandler',
                # 'formatter': 'pretty_traceback',
            },
        },
        'loggers': {
            'savant': {
                'level': log_level,
                'handlers': ['savantrs'],
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

    # install pretty traceback hook
    try:
        import pretty_traceback

        pretty_traceback.install()
    except ImportError:
        pass

    # add custom TRACE logging level
    add_logging_level('TRACE', logging.DEBUG - 5)

    level = get_default_loglevel() if log_level is None else log_level.upper()

    set_savant_rs_loglevel(level)

    log_config = get_log_conf(level)
    logging.config.dictConfig(log_config)
    init_logging.done = True


init_logging.done = False


def update_logging(log_level: str):
    """Update logging with specified log level.

    :param log_level: One of supported by logging module: INFO, DEBUG, etc.
    """
    log_level = log_level.upper()
    set_savant_rs_loglevel(log_level)
    logging.config.dictConfig(get_log_conf(log_level))



def set_savant_rs_loglevel(log_level:str):
    default_log_level_int = getattr(logging, get_default_loglevel())
    py_log_level_int = getattr(logging, log_level, default_log_level_int)

    rs_log_level = log_level_py_to_rs(py_log_level_int)

    # set savant_rs base logging level
    # no messages with priority lower than this setting are going to be logged
    # regardless of RUST_LOG env var config
    set_log_level(rs_log_level)
