from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ConfigKeyError

import importlib

from .schema import Action, Config, FlowConfig, PyFuncConfig, QueueConfig, WatchConfig

SECONDS_PER_UNIT = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400, 'w': 604800}


def convert_to_seconds(s: str):
    seconds = int(s[:-1]) * SECONDS_PER_UNIT[s[-1]]
    if seconds < 0:
        raise ValueError('Invalid input')
    return seconds


class ConfigParser:
    def __init__(self, config_path: str):
        self._config_path = config_path

    @staticmethod
    def __parse_labels(labels_list: ListConfig) -> list:
        container_labels = []
        for label_dict in OmegaConf.to_object(labels_list):
            labels = label_dict.get('labels')
            if labels is not None:
                if isinstance(labels, list):
                    container_labels.append(labels)
                else:
                    container_labels.append([labels])
        return container_labels

    @staticmethod
    def __parse_label_filters(config: DictConfig):
        label_filters = config.get('label_filters')
        if label_filters is None:
            return None
        return OmegaConf.to_object(label_filters)

    @staticmethod
    def __parse_queue_config(queue_config: DictConfig):
        if queue_config is None:
            return None

        return QueueConfig(
            action=Action(queue_config['action']),
            length=queue_config['length'],
            cooldown=convert_to_seconds(queue_config['cooldown']),
            polling_interval=convert_to_seconds(queue_config['polling_interval']),
            container_labels=ConfigParser.__parse_labels(queue_config['container']),
            label_filters=ConfigParser.__parse_label_filters(queue_config),
        )

    @staticmethod
    def __parse_flow_config(flow_config: DictConfig):
        if flow_config is None:
            return None

        idle = convert_to_seconds(flow_config['idle'])
        polling_interval = flow_config.get('polling_interval')

        return FlowConfig(
            action=Action(flow_config['action']),
            idle=idle,
            cooldown=convert_to_seconds(flow_config['cooldown']),
            polling_interval=(
                convert_to_seconds(polling_interval) if polling_interval else idle
            ),
            container_labels=ConfigParser.__parse_labels(flow_config['container']),
            label_filters=ConfigParser.__parse_label_filters(flow_config),
        )

    @staticmethod
    def __parse_pyfunc_config(pyfunc_config: DictConfig):
        if pyfunc_config is None:
            return None

        module_path = pyfunc_config['module']
        class_name = pyfunc_config['class_name']

        try:
            mod = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise ValueError(
                f'Failed to import pyfunc module "{module_path}": {e}'
            ) from e

        if not hasattr(mod, class_name):
            raise ValueError(
                f'Class "{class_name}" not found in module "{module_path}"'
            )

        kwargs = pyfunc_config.get('kwargs')
        if kwargs is not None:
            kwargs = OmegaConf.to_object(kwargs)

        return PyFuncConfig(
            action=Action(pyfunc_config['action']),
            cooldown=convert_to_seconds(pyfunc_config['cooldown']),
            polling_interval=convert_to_seconds(pyfunc_config['polling_interval']),
            container_labels=ConfigParser.__parse_labels(pyfunc_config['container']),
            module=module_path,
            class_name=class_name,
            kwargs=kwargs,
        )

    @staticmethod
    def __parse_watch_config(watch_config: DictConfig):
        return WatchConfig(
            buffer=watch_config['buffer'],
            queue=ConfigParser.__parse_queue_config(watch_config.get('queue')),
            egress=ConfigParser.__parse_flow_config(watch_config.get('egress')),
            ingress=ConfigParser.__parse_flow_config(watch_config.get('ingress')),
            pyfunc=ConfigParser.__parse_pyfunc_config(watch_config.get('pyfunc')),
        )

    def parse(self) -> Config:
        with open(self._config_path, 'r') as file:
            parsed_yaml = OmegaConf.load(file)
            watch = (
                parsed_yaml.get('watch')
                if isinstance(parsed_yaml, DictConfig)
                else None
            )

            if not watch:
                raise ValueError(
                    'No watch configs found in the config file. '
                    'Please specify at least one.'
                )

            try:
                config = Config([self.__parse_watch_config(w) for w in watch])
            except ConfigKeyError as e:
                raise ValueError(
                    f'Field "{e.key}" must be specified in the watch config.'
                )

        return config
