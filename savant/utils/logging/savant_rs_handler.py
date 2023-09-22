"""SavantRsLoggingHandler module."""
import logging

import pretty_traceback
from savant_rs.logging import LogLevel, log

LOG_LEVEL_PY_TO_RS = {
    logging.CRITICAL: LogLevel.Error,
    logging.ERROR: LogLevel.Error,
    logging.WARNING: LogLevel.Warning,
    logging.INFO: LogLevel.Info,
    logging.DEBUG: LogLevel.Debug,
}


class SavantRsLoggingHandler(logging.Handler):
    """Custom logging Handler that passes the log messages
    to the rust log from savant_rs with appropriate log level.

    No need to specify formatter during logging configuration:
    1. for all messages under ERROR priority no formatting is performed
       (rely on underlying rust log formatter)
    2. pretty_traceback formatter for ERROR and CRITICAL messages
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        self.pretty_trace_back_threshold = logging.ERROR
        self.formatter_pretty_traceback = pretty_traceback.LoggingFormatter()

    def format(self, record: logging.LogRecord) -> str:
        """Format a record."""
        if record.levelno < self.pretty_trace_back_threshold:
            return record.getMessage()
        return self.formatter_pretty_traceback.format(record)

    def emit(self, record: logging.LogRecord):
        """Emit a record."""
        try:
            formatted = self.format(record)
        except Exception:
            self.handleError(record)
            return

        log_level = LOG_LEVEL_PY_TO_RS[record.levelno]
        log(log_level, record.name, formatted)
