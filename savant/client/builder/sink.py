import logging
from typing import Optional

from savant.client.log_provider import LogProvider
from savant.client.runner.sink import SinkRunner

logger = logging.getLogger(__name__)


class SinkBuilder:
    """Builder for Sink.

    Usage example:

    .. code-block:: python

        sink = (
            SinkBuilder()
            .with_socket('rep+connect:ipc:///tmp/zmq-sockets/output-video.ipc')
            .with_idle_timeout(60)
            .with_log_provider(JaegerLogProvider('http://localhost:16686'))
            .build()
        )
        for result in sink:
            print(result.frame_meta)
            result.logs().pretty_print()
    """

    def __init__(
        self,
        socket: Optional[str] = None,
        log_provider: Optional[LogProvider] = None,
        idle_timeout: Optional[int] = None,
    ):
        self._socket = socket
        self._log_provider = log_provider
        self._idle_timeout = idle_timeout

    def with_socket(self, socket: str) -> 'SinkBuilder':
        """Set ZeroMQ socket for Sink."""
        return self._with_field('socket', socket)

    def with_log_provider(self, log_provider: LogProvider) -> 'SinkBuilder':
        """Set log provider for Sink."""
        return self._with_field('log_provider', log_provider)

    def with_idle_timeout(self, idle_timeout: Optional[int]) -> 'SinkBuilder':
        """Set idle timeout for Sink.

        Sink will stop trying to receive a message from ZeroMQ socket when it
        did not receive a message for idle_timeout seconds.
        """
        return self._with_field('idle_timeout', idle_timeout)

    def build(self) -> SinkRunner:
        """Build Sink."""

        assert self._socket is not None, 'socket is required'
        logger.debug(
            'Building source with socket %s and log provider %s.',
            self._socket,
            self._log_provider,
        )
        return SinkRunner(
            socket=self._socket,
            log_provider=self._log_provider,
            idle_timeout=self._idle_timeout,
        )

    def __repr__(self):
        return (
            f'SinkBuilder('
            f'socket={self._socket}, '
            f'log_provider={self._log_provider}, '
            f'idle_timeout={self._idle_timeout})'
        )

    def _with_field(self, field: str, value) -> 'SinkBuilder':
        return SinkBuilder(
            **{
                'socket': self._socket,
                'log_provider': self._log_provider,
                'idle_timeout': self._idle_timeout,
                field: value,
            }
        )
