"""Parameter storage package."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf
from savant_rs.video_object_query import (
    register_config_resolver,
    register_env_resolver,
    register_etcd_resolver,
    register_utility_resolver,
)

__all__ = ['param_storage', 'init_param_storage']

ParameterStorage = Dict[str, Any]
__PARAM_STORAGE: Optional[ParameterStorage] = None


@dataclass
class EtcdStorageConfig:
    """Etcd storage configuration parameters."""

    hosts: List[str]
    """The list of Etcd hosts to connect to. E.g. ["127.0.0.1:2379"]."""

    credentials: Optional[Tuple[str, str]] = None
    """The credentials to use for authentication."""

    watch_path: str = 'savant'
    """The path in Etcd used as the source of the state."""

    connect_timeout: int = 5
    """The timeout for connecting to Etcd."""

    watch_path_wait_timeout: int = 5
    """Waiting timeout for the watch path."""


def param_storage() -> ParameterStorage:
    """Get parameter storage.

    :return: Parameter storage.
    """
    return __PARAM_STORAGE


def init_param_storage(config: DictConfig) -> None:
    """Initialize parameter storage.

    :param config: Module config.
    """
    global __PARAM_STORAGE

    if __PARAM_STORAGE is not None:
        return

    __PARAM_STORAGE = {}

    # setup static parameters
    for name, value in config.parameters.items():
        __PARAM_STORAGE[name] = value

    # convert conf to dict with keys in dot notation to init config resolver
    def to_dot_keys(value: DictConfig) -> Dict[str, str]:
        def _to_dot_keys(value: Any, start_key: str, res: List[Tuple[str, str]]):
            if isinstance(value, (DictConfig, dict)):
                res += [
                    (f'{start_key}#type', 'dict'),
                    (f'{start_key}#len', str(len(value))),
                ]
                if value:
                    for key, val in value.items():
                        _to_dot_keys(val, f'{start_key}.{key}', res)
            elif isinstance(value, (ListConfig, list)):
                res += [
                    (f'{start_key}#type', 'list'),
                    (f'{start_key}#len', str(len(value))),
                ]
                if value:
                    for idx, val in enumerate(value):
                        _to_dot_keys(val, f'{start_key}[{idx}]', res)
            else:
                res += [
                    (f'{start_key}#type', type(value).__name__),
                    (start_key, str(value)),
                ]

        res = []
        for key, val in value.items():
            _to_dot_keys(val, key, res)
        return dict(res)

    # init resolvers
    register_config_resolver(to_dot_keys(config.parameters))
    register_env_resolver()
    register_utility_resolver()

    if 'etcd' in config.parameters and config.parameters.etcd:
        storage_config_schema = OmegaConf.structured(EtcdStorageConfig)
        storage_config = OmegaConf.unsafe_merge(
            storage_config_schema, config.parameters.etcd
        )
        register_etcd_resolver(**storage_config)
