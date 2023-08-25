import logging
from typing import Optional

from savant.client.log_provider import LogProvider
from savant.client.runner.source import SourceRunner

logger = logging.getLogger(__name__)


class SourceBuilder:
    def __init__(
        self,
        socket: Optional[str] = None,
        log_provider: Optional[LogProvider] = None,
        retries: int = 3,
    ):
        self._socket = socket
        self._log_provider = log_provider
        self._retries = retries

    def with_socket(self, socket: str) -> 'SourceBuilder':
        return self._with_field('socket', socket)

    def with_log_provider(self, log_provider) -> 'SourceBuilder':
        return self._with_field('log_provider', log_provider)

    def with_retries(self, retries: int) -> 'SourceBuilder':
        return self._with_field('retries', retries)

    def build(self) -> SourceRunner:
        assert self._socket is not None, 'socket is required'
        logger.debug(
            'Building source with socket %s and log provider %s.',
            self._socket,
            self._log_provider,
        )
        return SourceRunner(
            socket=self._socket,
            log_provider=self._log_provider,
            retries=self._retries,
        )

    def __repr__(self):
        return (
            f'SourceBuilder('
            f'socket={self._socket}, '
            f'log_provider={self._log_provider},'
            f'retries={self._retries})'
        )

    def _with_field(self, field: str, value) -> 'SourceBuilder':
        return SourceBuilder(
            **{
                'socket': self._socket,
                'log_provider': self._log_provider,
                'retries': self._retries,
                field: value,
            }
        )
