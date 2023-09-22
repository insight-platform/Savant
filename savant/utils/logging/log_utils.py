"""General logging utils."""
import logging
import os
import string

from savant_rs.logging import LogLevel, set_log_level

from .const import DEFAULT_LOGLEVEL, LOGGING_PREFIX

LOG_LEVEL_STR_TO_RS = {
    'error': LogLevel.Error,
    'warn': LogLevel.Warning,
    'info': LogLevel.Info,
    'debug': LogLevel.Debug,
    'trace': LogLevel.Trace,
}


def parse_log_spec(log_spec_str: str) -> dict:
    """Parse logging specification string.

    :param log_spec_str: A comma-separated list of logging directives
        of the form target=level.
    :return:  A dictionary of the form log_target:log_level.
    """
    log_spec_str = log_spec_str.strip(string.whitespace + ',')
    log_spec_str = log_spec_str.replace('::', '.')
    log_spec_dict = {}
    for log_directive in log_spec_str.split(','):
        # consecutive commas will result in empty strings
        # skip them
        if log_directive:

            eq_num = log_directive.count('=')

            if eq_num == 1:
                target, level = log_directive.split('=')
            elif eq_num == 0:
                log_directive = log_directive.lower()
                # no = sign, i.e. only target or only level
                if log_directive in LOG_LEVEL_STR_TO_RS:
                    # only level
                    level = log_directive
                    target = LOGGING_PREFIX
                else:
                    # only target
                    target = log_directive
                    level = DEFAULT_LOGLEVEL
            else:
                # more than one = sign, incorrect string
                # skip this target_level
                continue
            log_spec_dict[target] = level.lower()

    return log_spec_dict


def get_default_log_spec() -> dict:
    """Get default logging specification."""
    log_spec_str = os.environ.get('LOGLEVEL', DEFAULT_LOGLEVEL)
    return parse_log_spec(log_spec_str)


def set_savant_rs_loglevel(log_spec_dict: dict):
    """Set savant_rs base logging level.
    No messages with priority lower than this setting are going to be logged.

    :param log_spec_dict: A dictionary of the form log_target:log_level.
    """

    log_level_order = ['trace', 'debug', 'info', 'warn', 'error']

    def get_lowest_level():
        for log_level in log_level_order:
            if log_level in log_spec_dict.values():
                return log_level
        return DEFAULT_LOGLEVEL

    rs_log_level = LOG_LEVEL_STR_TO_RS[get_lowest_level()]
    set_log_level(rs_log_level)


def get_log_conf(log_spec_dict: dict) -> dict:
    """Create logging configuration for use in logging.config.dictConfig().

    :param log_spec_dict: A dictionary of the form log_target:log_level.
    :return: Logging configuration dictionary.
    """
    main_level = log_spec_dict.pop(LOGGING_PREFIX, DEFAULT_LOGLEVEL)
    loggers = {LOGGING_PREFIX: {'level': main_level.upper(), 'handlers': ['savantrs']}}

    for target, level in log_spec_dict.items():
        if target.startswith(LOGGING_PREFIX):
            handlers = []
        else:
            handlers = ['savantrs']
        loggers[target] = {
            'level': level.upper(),
            'handlers': handlers,
        }

    return {
        'version': 1,
        'formatters': {},
        'handlers': {
            'savantrs': {
                'class': 'savant.utils.logging.savant_rs_handler.SavantRsLoggingHandler',
            },
        },
        'loggers': loggers,
    }


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
