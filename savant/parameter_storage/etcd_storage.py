"""Etcd storage module."""
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union
import json
import logging
from omegaconf import MISSING
import etcd3
from .parameter_storage import ParameterStorage
from savant.utils.logging import get_logger
logger = get_logger(__name__)


@dataclass
class EtcdStorageEndpoint:
    # pylint:disable=line-too-long
    """https://github.com/kragniz/python-
    etcd3/blob/e78000e00a25d2c3e23dbdb078eb7e20e6f5fe3d/etcd3/client.py#L72."""

    host: str
    port: str
    secure: bool = False
    # TODO: Credentials to use for secure channel, required if secure = True
    #  Has to be tested on cloud installation
    creds: Any = None


@dataclass
class EtcdStorageConfig:
    # pylint:disable=line-too-long
    """https://github.com/kragniz/python-etcd3/blob/e78000e00a25d2c3e23dbdb078e
    b7e20e6f5fe3d/etcd3/client.py#L145."""

    endpoints: List[EtcdStorageEndpoint] = MISSING
    timeout: Optional[float] = None
    failover: bool = True


class EtcdStorage(ParameterStorage):
    """Manages cached access to value from etcd storage."""

    def __init__(self, **kwargs) -> None:
        """init Tries to create etcd client."""

        self._vals = {}
        self._watch_ids = {}

        try:
            kwargs['endpoints'] = [
                etcd3.Endpoint(**endpoint_kwargs)
                for endpoint_kwargs in kwargs['endpoints']
            ]
            self._etcd = etcd3.MultiEndpointEtcd3Client(**kwargs)
        except etcd3.Etcd3Exception:
            # retry later?
            logger.exception('Etcd client init failed.')
            self._etcd = None

    def __getitem__(self, name: str) -> Any:
        """Get value for a given name.

        Name is expected to be registered first.
        """

        return self._vals[name]

    def __setitem__(self, name: str, value: Any):
        """Set value for a given name."""
        self._vals[name] = value

    def register_parameter(
        self, name: str, default_value: Optional[Any] = None
    ) -> None:
        """Registers static parameter.

        :param name: Parameter name.
        :param default_value: Default value for a given name.
        """
        self._register_parameter(name, default_value)

    def register_dynamic_parameter(
        self,
        name: str,
        default_value: Optional[Any] = None,
        on_change: Optional[Callable] = None,
    ) -> None:
        """Prepare the manager to be able to return some value for a given
        name. Try to get value from storage, fallback to default in case of
        fail. Try to add watch for name.

        :param name: Parameter name in etcd.
        :param default_value: Default value for a given name,
            if skipped and storage request failed then subsequent __getitem__
            calls may fail.
        :param on_change: Additional user specified action
            to perform on decoded event value from watch response.
        """
        self._register_parameter(name, default_value)

        if name not in self._watch_ids:
            try:
                self._add_watch(name, on_change)
            except (etcd3.Etcd3Exception, TypeError):
                # retry later?
                logger.exception('Watch add for "%s" failed.', name)

    def _register_parameter(self, name: str, default_value: Any) -> None:
        if name not in self._vals:
            try:
                val, _ = self._etcd.get(name)
                val = EtcdStorage._decode_value(val)
            except (
                etcd3.Etcd3Exception,
                TypeError,
                json.decoder.JSONDecodeError,
            ) as exc:
                # retry later?
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception(
                        'Getting parameter "%s" from storage failed.', name
                    )
                else:
                    logger.error(
                        'Getting parameter "%s" from storage failed. Reason: "%s".',
                        name,
                        exc,
                    )
                val = default_value

            if val is not None:
                self._vals[name] = val

    def _add_watch(self, name: str, on_change: Optional[Callable] = None):
        """Etcd add watch request."""

        def watch_callback(watch_response):
            for event in watch_response.events:
                if isinstance(event, etcd3.events.PutEvent):
                    try:
                        new_val = EtcdStorage._decode_value(event.value)
                        self._vals[name] = new_val
                        if on_change is not None:
                            on_change(new_val)
                    except (json.decoder.JSONDecodeError, TypeError):
                        logger.exception(
                            'New value from storage for "%s" is incorrect, '
                            'update failed, will leave parameter unchanged.'
                        )
                elif isinstance(event, etcd3.events.DeleteEvent):
                    # is this an important case?
                    logger.debug('Received DeleteEvent from storage for "%s".', name)

        watch_id = self._etcd.add_watch_callback(name, watch_callback)
        self._watch_ids[name] = watch_id

    @staticmethod
    def _decode_value(val: Union[str, bytes]):
        """Decode config value."""

        return json.loads(val)

    def __del__(self):
        """Cancel watches."""

        for name, watch_id in self._watch_ids.items():
            try:
                self._etcd.cancel_watch(watch_id)
            except etcd3.Etcd3Exception:
                logger.exception('Cancel watch for "%s" failed.', name)
