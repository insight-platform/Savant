import logging
from savant_rs.logging import log
from .log_utils import log_level_py_to_rs

class SavantRsLoggingHandler(logging.Handler):
    def __init__(self) -> None:
        logging.Handler.__init__(self)

    def emit(self, record):
        record_name_rs = record.name.replace('.', '::')
        log_level = log_level_py_to_rs(record.levelno)
        msg = record.getMessage()
        log(log_level, record_name_rs, msg)
