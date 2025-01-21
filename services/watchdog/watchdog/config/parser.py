from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError

from .schema import Action, Config, FlowConfig, QueueConfig, WatchConfig

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
    def __parse_labels(labels_list: list) -> list:
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
    def __parse_queue_config(queue_config: dict):
        if queue_config is None:
            return None

        return QueueConfig(
            action=Action(queue_config['action']),
            length=queue_config['length'],
            cooldown=convert_to_seconds(queue_config['cooldown']),
            polling_interval=convert_to_seconds(queue_config['polling_interval']),
            container_labels=ConfigParser.__parse_labels(queue_config['container']),
        )

    @staticmethod
    def __parse_flow_config(flow_config: dict):
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
        )

    @staticmethod
    def __parse_watch_config(watch_config: dict):
        return WatchConfig(
            buffer=watch_config['buffer'],
            queue=ConfigParser.__parse_queue_config(watch_config.get('queue')),
            egress=ConfigParser.__parse_flow_config(watch_config.get('egress')),
            ingress=ConfigParser.__parse_flow_config(watch_config.get('ingress')),
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
                    'No watch configs found in the config file. Please specify at least one.'
                )

            try:
                config = Config([self.__parse_watch_config(w) for w in watch])
            except ConfigKeyError as e:
                raise ValueError(
                    f'Field "{e.key}" must be specified in the watch config.'
                )

        return config
