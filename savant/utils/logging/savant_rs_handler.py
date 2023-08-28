"""SavantRsLoggingHandler module."""
import logging

import pretty_traceback
from savant_rs.logging import log

from .log_utils import log_level_py_to_rs


class SavantRsLoggingHandler(logging.Handler):
    """Custom logging Handler that passes the log messages
    to the rust log from savant_rs with appropriate log level.

    Hardcoded to use two formatters (no need to specify during logging configuration):
    1. basic formatter for all messages under ERROR priority
    2. pretty_traceback formatter for ERROR and CRITICAL messages
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        self.pretty_trace_back_threshold = logging.ERROR
        self.formatter_basic = logging.Formatter()
        self.formatter_pretty_traceback = pretty_traceback.LoggingFormatter()

    def format(self, record: logging.LogRecord) -> str:
        """Format a record."""
        if record.levelno < self.pretty_trace_back_threshold:
            return self.formatter_basic.format(record)
        return self.formatter_pretty_traceback.format(record)

    def emit(self, record: logging.LogRecord):
        """Emit a record."""
        try:
            formatted = self.format(record)
        except Exception:
            self.handleError(record)
            return

        log_level = log_level_py_to_rs(record.levelno)
        log(log_level, record.name, formatted)
