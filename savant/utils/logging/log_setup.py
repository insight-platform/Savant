"""Logging setup utils."""

import logging
import logging.config
from typing import Optional

from savant_rs.logging import LogLevel

from .const import LOGGING_PREFIX
from .log_utils import (
    add_logging_level,
    get_default_log_spec,
    get_log_conf,
    parse_log_spec,
    set_savant_rs_loglevel,
)
from .savant_rs_handler import LOG_LEVEL_PY_TO_RS


def init_logging(log_spec_str: Optional[str] = None):
    """Initialize logging with specified log level or set default.

    :param log_spec_str: A comma-separated list of logging directives of the form target=level.
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
    LOG_LEVEL_PY_TO_RS[logging.TRACE] = LogLevel.Trace

    if log_spec_str is None:
        log_spec_dict = get_default_log_spec()
    else:
        log_spec_dict = parse_log_spec(log_spec_str)

    apply_log_spec(log_spec_dict)

    init_logging.done = True


init_logging.done = False


def get_logger(name: str) -> logging.Logger:
    """Get logger with specified name appended to the general Savant logging prefix.

    :param name: Logger name.
    :return: Logger instance.
    """
    # prevent modules from the main savant package having savant.savant prefix
    logger_name = '.'.join((LOGGING_PREFIX, name)).replace('savant.savant.', 'savant.')
    return logging.getLogger(logger_name)


def update_logging(log_spec_str: str):
    """Update logging with specified log spec.

    :param log_spec_str: A comma-separated list of logging directives of the form target=level.
    """
    log_spec_dict = parse_log_spec(log_spec_str)
    apply_log_spec(log_spec_dict)


def apply_log_spec(log_spec_dict: dict):
    """Apply log specification.

    :param log_spec_dict: A dictionary of of the form log_target:log_level.
    """
    set_savant_rs_loglevel(log_spec_dict)
    log_config = get_log_conf(log_spec_dict)
    logging.config.dictConfig(log_config)
