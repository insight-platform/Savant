import os


def opt_config(name, default=None, convert=None):
    conf_str = os.environ.get(name)
    if conf_str:
        return convert(conf_str) if convert else conf_str
    return default
