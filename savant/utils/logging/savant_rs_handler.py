import logging
from savant_rs.logging import log, LogLevel

class SavantRsLoggingHandler(logging.Handler):
    def __init__(self) -> None:
        logging.Handler.__init__(self)
    def emit(self, record):
        record_name_rs = record.name.replace('.', '::')
        if record.levelno in (logging.ERROR, logging.CRITICAL):
            log_level = LogLevel.Error
        elif record.levelno == logging.WARNING:
            log_level = LogLevel.Warning
        elif record.levelno == logging.INFO:
            log_level = LogLevel.Info
        elif record.levelno == logging.DEBUG:
            log_level = LogLevel.Debug
        else:
            log_level = LogLevel.Trace
        msg = record.getMessage()
        log(log_level, record_name_rs, msg)
