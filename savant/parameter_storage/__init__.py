"""Parameter storage package."""

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf
from savant_rs.match_query import (
    EtcdCredentials,
    TlsConfig,
    register_config_resolver,
    register_env_resolver,
    register_etcd_resolver,
    register_utility_resolver,
)

from savant.utils.logging import get_logger

__all__ = ['param_storage', 'init_param_storage']

ParameterStorage = Dict[str, Any]
__PARAM_STORAGE: Optional[ParameterStorage] = None
logger = get_logger(__name__)


@dataclass
class EtcdCredentialsConfig:
    """Etcd credentials."""

    username: str
    """The username."""

    password: str
    """The password."""


@dataclass
class TlsParametersConfig:
    """TLS configuration parameters."""

    ca_certificate: pathlib.Path
    """The certificate authority file path."""

    certificate: pathlib.Path
    """The client certificate file path."""

    key: pathlib.Path
    """The client key file path."""


@dataclass
class EtcdStorageConfig:
    """Etcd storage configuration parameters."""

    hosts: List[str]
    """The list of Etcd hosts to connect to. E.g. ["127.0.0.1:2379"]."""

    credentials: Optional[EtcdCredentialsConfig] = None
    """The credentials to use for authentication."""

    tls: Optional[TlsParametersConfig] = None
    """The TLS configuration."""

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
        storage_config: EtcdStorageConfig = OmegaConf.unsafe_merge(
            storage_config_schema, config.parameters.etcd
        )
        hosts: List[str] = storage_config.hosts
        credentials: EtcdCredentials = None
        if storage_config.credentials:
            credentials = EtcdCredentials(
                storage_config.credentials.username, storage_config.credentials.password
            )
        tls_config: TlsConfig = None
        if storage_config.tls:
            try:
                logger.info(
                    'Loading Etcd CA from %s.', storage_config.tls.ca_certificate
                )
                ca_certificate = storage_config.tls.ca_certificate.read_text()
                logger.info(
                    'Loading Etcd client cert from %s.', storage_config.tls.certificate
                )
                cert = storage_config.tls.certificate.read_text()
                logger.info('Loading Etcd client key from %s.', storage_config.tls.key)
                key = storage_config.tls.key.read_text()
                tls_config = TlsConfig(ca_certificate, cert, key)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f'File not found when loading Etcd CA, cert or key: {exc.filename}.'
                    ' Please check the path to the file.'
                )

        watch_path: str = storage_config.watch_path
        connect_timeout: int = storage_config.connect_timeout
        watch_path_wait_timeout: int = storage_config.watch_path_wait_timeout

        register_etcd_resolver(
            hosts,
            credentials,
            tls_config,
            watch_path,
            connect_timeout,
            watch_path_wait_timeout,
        )
