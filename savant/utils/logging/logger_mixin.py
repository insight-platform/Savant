"""LoggerMixin module."""
import logging

from .log_setup import init_logging


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

        init_logging()
        self._logger = logging.getLogger(logger_name)

        self._logger_initialized = True
