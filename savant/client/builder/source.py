from typing import Optional

from savant.client.log_provider import LogProvider
from savant.client.runner.source import AsyncSourceRunner, SourceRunner
from savant.utils.logging import get_logger

logger = get_logger(__name__)


class SourceBuilder:
    """Builder for Source.

    Usage example:

    .. code-block:: python

        source = (
            SourceBuilder()
            .with_log_provider(JaegerLogProvider('http://localhost:16686'))
            .with_socket('req+connect:ipc:///tmp/zmq-sockets/input-video.ipc')
            .build()
        )
        result = source(JpegSource('cam-1', 'data/AVG-TownCentre.jpeg'))
        result.logs().pretty_print()

    Usage example (async):

    .. code-block:: python

        source = (
            SourceBuilder()
            .with_log_provider(JaegerLogProvider('http://localhost:16686'))
            .with_socket('req+connect:ipc:///tmp/zmq-sockets/input-video.ipc')
            .build_async()
        )
        result = await source(JpegSource('cam-1', 'data/AVG-TownCentre.jpeg'))
        result.logs().pretty_print()
    """

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
        """Set ZeroMQ socket for Source."""
        return self._with_field('socket', socket)

    def with_log_provider(self, log_provider) -> 'SourceBuilder':
        """Set log provider for Source."""
        return self._with_field('log_provider', log_provider)

    def with_retries(self, retries: int) -> 'SourceBuilder':
        """Set number of retries for Source.

        Source retries to send message to ZeroMQ socket if it fails.
        """
        return self._with_field('retries', retries)

    def build(self) -> SourceRunner:
        """Build Source."""

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

    def build_async(self) -> AsyncSourceRunner:
        """Build async Source."""

        assert self._socket is not None, 'socket is required'
        logger.debug(
            'Building async source with socket %s and log provider %s.',
            self._socket,
            self._log_provider,
        )
        return AsyncSourceRunner(
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
