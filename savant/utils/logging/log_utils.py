"""Logger utils."""
import logging
import logging.config
from savant_rs.logging import LogLevel

def add_logging_level(
    level_name, level_num, method_name=None, *, exc_info=False, stack_info=False
):

    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    Based on https://haggis.readthedocs.io/en/stable/api.html#haggis.logs.add_logging_level
    """

    def for_logger_adapter(self, msg, *args, **kwargs):
        kwargs.setdefault('exc_info', exc_info)
        kwargs.setdefault('stack_info', stack_info)
        self.log(level_num, msg, *args, **kwargs)

    def for_logger_class(self, msg, *args, **kwargs):
        if self.isEnabledFor(level_num):
            kwargs.setdefault('exc_info', exc_info)
            kwargs.setdefault('stack_info', stack_info)
            self._log(level_num, msg, args, **kwargs)

    def for_logging_module(*args, **kwargs):
        kwargs.setdefault('exc_info', exc_info)
        kwargs.setdefault('stack_info', stack_info)
        logging.log(level_num, *args, **kwargs)

    if not method_name:
        method_name = level_name.lower()
    if method_name == level_name:
        raise ValueError('Method name must differ from level name')

    def check_conflict(conflict, message):
        if conflict:
            raise AttributeError(message)
        return conflict

    def check_func_conflict(func, name, original_name, is_func, target):
        conflict = not (
            callable(func)
            and getattr(func, '_original_name', None) == original_name
            and getattr(func, '_exc_info', None) == exc_info
            and getattr(func, '_stack_info', None) == stack_info
        )
        return check_conflict(
            conflict,
            '{} {!r} already defined in {}'.format(
                'Function' if is_func else 'Method', name, target
            ),
        )

    logging._acquireLock()
    try:
        registered_num = logging.getLevelName(level_name)
        logger_class = logging.getLoggerClass()
        logger_adapter = logging.LoggerAdapter

        if registered_num != 'Level ' + level_name:
            check_conflict(
                registered_num != level_num,
                'Level {!r} already registered ' 'in logging module'.format(level_name),
            )

        current_level = getattr(logging, level_name, None)
        if current_level is not None:
            check_conflict(
                current_level != level_num,
                'Level {!r} already defined ' 'in logging module'.format(level_name),
            )

        logging_func = getattr(logging, method_name, None)
        if logging_func is not None:
            check_func_conflict(
                logging_func,
                method_name,
                for_logging_module.__name__,
                True,
                'logging module',
            )

        logger_method = getattr(logger_class, method_name, None)
        if logger_method is not None:
            check_func_conflict(
                logger_method,
                method_name,
                for_logger_class.__name__,
                False,
                'logger class',
            )

        adapter_method = getattr(logger_adapter, method_name, None)
        if adapter_method is not None:
            check_func_conflict(
                adapter_method,
                method_name,
                for_logger_adapter.__name__,
                False,
                'logger adapter',
            )

        # Make sure the method names are set to sensible values, but
        # preserve the names of the old methods for future verification.
        def label_func(func):
            func._original_name = func.__name__
            func.__name__ = method_name
            func._exc_info = exc_info
            func._stack_info = stack_info

        label_func(for_logging_module)
        label_func(for_logger_class)
        label_func(for_logger_adapter)

        # Actually add the new level
        logging.addLevelName(level_num, level_name)
        setattr(logging, level_name, level_num)
        setattr(logging, method_name, for_logging_module)
        setattr(logger_class, method_name, for_logger_class)
        setattr(logger_adapter, method_name, for_logger_adapter)
    finally:
        logging._releaseLock()

def log_level_py_to_rs(py_log_level: int):
    if py_log_level in (logging.ERROR, logging.CRITICAL):
        return LogLevel.Error
    if py_log_level == logging.WARNING:
        return LogLevel.Warning
    if py_log_level == logging.INFO:
        return LogLevel.Info
    if py_log_level == logging.DEBUG:
        return LogLevel.Debug
    if py_log_level == logging.TRACE:
        return LogLevel.Trace
    raise AttributeError(f'No rust pair for py log level {py_log_level}')
