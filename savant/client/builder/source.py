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
            # Note: healthcheck port should be configured in the module.
            .with_module_health_check_url('http://172.17.0.1:8888/healthcheck')
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
        module_health_check_url: Optional[str] = None,
        module_health_check_timeout: float = 60,
        module_health_check_interval: float = 5,
        telemetry_enabled: bool = True,
    ):
        self._socket = socket
        self._log_provider = log_provider
        self._retries = retries
        self._module_health_check_url = module_health_check_url
        self._module_health_check_timeout = module_health_check_timeout
        self._module_health_check_interval = module_health_check_interval
        self._telemetry_enabled = telemetry_enabled

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

    def with_module_health_check_url(self, url: str) -> 'SourceBuilder':
        """Set module health check url for Source.

        Source will check the module health before receiving any messages.
        """
        return self._with_field('module_health_check_url', url)

    def with_module_health_check_timeout(self, timeout: float) -> 'SourceBuilder':
        """Set module health check timeout for Source.

        Source will wait for the module to be ready for the specified timeout.
        """
        return self._with_field('module_health_check_timeout', timeout)

    def with_module_health_check_interval(self, interval: float) -> 'SourceBuilder':
        """Set module health check interval for Source.

        Source will check the module health every specified interval.
        """
        return self._with_field('module_health_check_interval', interval)

    def with_telemetry_enabled(self) -> 'SourceBuilder':
        """Enable telemetry for Source."""
        return self._with_field('telemetry_enabled', True)

    def with_telemetry_disabled(self) -> 'SourceBuilder':
        """Disable telemetry for Source."""
        return self._with_field('telemetry_enabled', False)

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
            module_health_check_url=self._module_health_check_url,
            module_health_check_timeout=self._module_health_check_timeout,
            module_health_check_interval=self._module_health_check_interval,
            telemetry_enabled=self._telemetry_enabled,
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
            module_health_check_url=self._module_health_check_url,
            module_health_check_timeout=self._module_health_check_timeout,
            module_health_check_interval=self._module_health_check_interval,
            telemetry_enabled=self._telemetry_enabled,
        )

    def __repr__(self):
        return (
            f'SourceBuilder('
            f'socket={self._socket}, '
            f'log_provider={self._log_provider},'
            f'retries={self._retries},'
            f'module_health_check_url={self._module_health_check_url}, '
            f'module_health_check_timeout={self._module_health_check_timeout}, '
            f'module_health_check_interval={self._module_health_check_interval}, '
            f'telemetry_enabled={self._telemetry_enabled})'
        )

    def _with_field(self, field: str, value) -> 'SourceBuilder':
        return SourceBuilder(
            **{
                'socket': self._socket,
                'log_provider': self._log_provider,
                'retries': self._retries,
                'module_health_check_url': self._module_health_check_url,
                'module_health_check_timeout': self._module_health_check_timeout,
                'module_health_check_interval': self._module_health_check_interval,
                'telemetry_enabled': self._telemetry_enabled,
                field: value,
            }
        )
