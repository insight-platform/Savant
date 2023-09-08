import os


def opt_config(name, default=None, convert=None):
    """Get an optional configuration value from the environment.

    :param name: The name of the environment variable.
    :param default: The default value to return if the configuration variable is not set.
    :param convert: A function to convert the value from string to a different type.
    """

    conf_str = os.environ.get(name)
    if conf_str:
        return convert(conf_str) if convert else conf_str
    return default
