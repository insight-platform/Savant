"""Parameter storage package."""
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from .etcd_storage import EtcdStorage, EtcdStorageConfig
from .parameter_storage import ParameterStorage

__all__ = ['param_storage', 'init_param_storage', 'STORAGE_TYPES']

__PARAM_STORAGE: Optional[ParameterStorage] = None
STORAGE_TYPES = frozenset(['etcd'])


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
    if __PARAM_STORAGE is None:
        storage_type = config.parameters.get('dynamic_parameter_storage', None)
        storage_cfg = config.parameters.get(f'{storage_type}_config', {})
        if storage_type == 'etcd':
            storage_config_schema = OmegaConf.structured(EtcdStorageConfig)
            storage_cfg = OmegaConf.unsafe_merge(storage_config_schema, storage_cfg)
            __PARAM_STORAGE = EtcdStorage(**storage_cfg)

        # setup static parameters
        for name, value in config.parameters.items():
            __PARAM_STORAGE[name] = value

        # register dynamic parameters
        for name, default in config.dynamic_parameters.items():
            __PARAM_STORAGE.register_dynamic_parameter(name, default)
