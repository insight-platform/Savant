import logging
from typing import Optional

from savant.source_sink_framework.log_provider import LogProvider
from savant.source_sink_framework.runner.source import SourceRunner

logger = logging.getLogger(__name__)


class SourceBuilder:
    def __init__(
        self,
        timeout: float = 0,
        socket: Optional[str] = None,
        log_provider: Optional[LogProvider] = None,
    ):
        self._timeout = timeout
        self._socket = socket
        self._log_provider = log_provider

    def with_timeout(self, timeout: float) -> 'SourceBuilder':
        return SourceBuilder(
            timeout=timeout,
            socket=self._socket,
            log_provider=self._log_provider,
        )

    def with_socket(self, socket: str) -> 'SourceBuilder':
        return SourceBuilder(
            timeout=self._timeout,
            socket=socket,
            log_provider=self._log_provider,
        )

    def with_log_provider(self, log_provider) -> 'SourceBuilder':
        return SourceBuilder(
            timeout=self._timeout,
            socket=self._socket,
            log_provider=log_provider,
        )

    def build(self) -> SourceRunner:
        assert self._socket is not None, 'socket is required'
        logger.debug(
            'Building source with socket %s, timeout %s and log provider %s.',
            self._socket,
            self._timeout,
            self._log_provider,
        )
        return SourceRunner(
            timeout=self._timeout,
            socket=self._socket,
            log_provider=self._log_provider,
        )

    def __repr__(self):
        return (
            f'SourceBuilder('
            f'timeout={self._timeout}, '
            f'socket={self._socket}, '
            f'log_provider={self._log_provider})'
        )
