import logging
import pretty_traceback
from savant_rs.logging import log
from .log_utils import log_level_py_to_rs

class SavantRsLoggingHandler(logging.Handler):
    def __init__(self) -> None:
        logging.Handler.__init__(self)
        self.pretty_trace_back_threshold = logging.ERROR
        self.formatter_basic = logging.Formatter()
        self.formatter_pretty_traceback = pretty_traceback.LoggingFormatter()

    def format(self, record):
        """Format a record."""
        if record.levelno < self.pretty_trace_back_threshold:
            return self.formatter_basic.format(record)
        return self.formatter_pretty_traceback.format(record)

    def emit(self, record):

        try:
            formatted = self.format(record)
        except Exception:
            self.handleError(record)
            return

        record_name_rs = record.name.replace('.', '::')
        log_level = log_level_py_to_rs(record.levelno)
        log(log_level, record_name_rs, formatted)
