import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from savant_rs.primitives import EndOfStream, VideoFrame, VideoFrameContent

from savant.client.log_provider import LogProvider
from savant.client.runner import LogResult
from savant.client.runner.healthcheck import HealthCheck
from savant.healthcheck.status import ModuleStatus
from savant.utils.logging import get_logger
from savant.utils.zeromq import AsyncZeroMQSource, Defaults, ZeroMQMessage, ZeroMQSource

logger = get_logger(__name__)


@dataclass
class SinkResult(LogResult):
    """Result of receiving a message from ZeroMQ socket."""

    frame_meta: Optional[VideoFrame]
    """Video frame metadata."""
    frame_content: Optional[bytes] = field(repr=False)
    """Video frame content."""
    eos: Optional[EndOfStream]
    """End of stream."""


class BaseSinkRunner(ABC):
    _source: ZeroMQSource

    def __init__(
        self,
        socket: str,
        log_provider: Optional[LogProvider],
        idle_timeout: Optional[int],
        module_health_check_url: Optional[str],
        module_health_check_timeout: float,
        module_health_check_interval: float,
        receive_timeout: int = Defaults.RECEIVE_TIMEOUT,
        receive_hwm: int = Defaults.RECEIVE_HWM,
    ):
        self._log_provider = log_provider
        self._idle_timeout = idle_timeout if idle_timeout is not None else 10**6
        self._health_check = (
            HealthCheck(
                url=module_health_check_url,
                interval=module_health_check_interval,
                timeout=module_health_check_timeout,
                ready_statuses=[ModuleStatus.RUNNING, ModuleStatus.STOPPING],
            )
            if module_health_check_url is not None
            else None
        )
        self._source = self._build_zeromq_source(socket, receive_timeout, receive_hwm)
        self._source.start()

    def __del__(self):
        self._source.terminate()

    @abstractmethod
    def _build_zeromq_source(self, socket: str, receive_timeout: int, receive_hwm: int):
        pass

    @abstractmethod
    def _receive_next_message(self) -> Optional[SinkResult]:
        pass

    def _handle_message(self, zmq_message: ZeroMQMessage):
        message = zmq_message.message
        message.validate_seq_id()
        trace_id: Optional[str] = message.span_context.as_dict().get('uber-trace-id')
        if trace_id is not None:
            trace_id = trace_id.split(':', 1)[0]
        if message.is_video_frame():
            video_frame: VideoFrame = message.as_video_frame()
            logger.debug(
                'Received video frame %s/%s.',
                video_frame.source_id,
                video_frame.pts,
            )
            content = zmq_message.content
            if not zmq_message.content:
                content = None
                if video_frame.content.is_internal():
                    content = video_frame.content.get_data_as_bytes()
                    video_frame.content = VideoFrameContent.none()
            return SinkResult(
                frame_meta=video_frame,
                frame_content=content,
                eos=None,
                trace_id=trace_id,
                log_provider=self._log_provider,
            )

        if message.is_end_of_stream():
            eos: EndOfStream = message.as_end_of_stream()
            logger.debug('Received EOS from source %s.', eos.source_id)
            return SinkResult(
                frame_meta=None,
                frame_content=None,
                eos=eos,
                trace_id=trace_id,
                log_provider=self._log_provider,
            )

        raise Exception('Unknown message type')


class SinkRunner(BaseSinkRunner):
    """Receives messages from ZeroMQ socket."""

    def _build_zeromq_source(self, socket: str, receive_timeout: int, receive_hwm: int):
        return ZeroMQSource(
            socket=socket,
            receive_timeout=receive_timeout,
            receive_hwm=receive_hwm,
            set_ipc_socket_permissions=False,
        )

    def __next__(self) -> SinkResult:
        """Receive next message from ZeroMQ socket.

        :return: Result of receiving a message from ZeroMQ socket.
        :raise StopIteration: If no message was received for idle_timeout seconds.
        """

        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        wait_until = time.time() + self._idle_timeout
        result = None
        while result is None:
            result = self._receive_next_message()
            if result is None and time.time() > wait_until:
                raise StopIteration()

        return result

    def __iter__(self):
        """Receive messages from ZeroMQ socket infinitely or until no message
        was received for idle_timeout seconds."""
        return self

    def _receive_next_message(self) -> Optional[SinkResult]:
        message = self._source.next_message()
        if message is not None:
            return self._handle_message(message)


class AsyncSinkRunner(BaseSinkRunner):
    """Receives messages from ZeroMQ socket asynchronously."""

    _source: AsyncZeroMQSource

    def _build_zeromq_source(self, socket: str, receive_timeout: int, receive_hwm: int):
        return AsyncZeroMQSource(
            socket=socket,
            receive_timeout=receive_timeout,
            receive_hwm=receive_hwm,
            set_ipc_socket_permissions=False,
        )

    async def __anext__(self) -> SinkResult:
        """Receive next message from ZeroMQ socket.

        :return: Result of receiving a message from ZeroMQ socket.
        :raise StopIteration: If no message was received for idle_timeout seconds.
        """

        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        wait_until = time.time() + self._idle_timeout
        result = None
        while result is None:
            result = await self._receive_next_message()
            if result is None and time.time() > wait_until:
                raise StopAsyncIteration()

        return result

    def __aiter__(self):
        """Receive messages from ZeroMQ socket infinitely or until no message
        was received for idle_timeout seconds."""
        return self

    async def _receive_next_message(self) -> Optional[SinkResult]:
        message = await self._source.next_message()
        if message is not None:
            return self._handle_message(message)
