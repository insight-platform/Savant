from dataclasses import dataclass
from typing import AsyncIterable, Iterable, List, Optional, Set, Tuple, Union

import zmq
from savant_rs.pipeline2 import VideoPipeline, VideoPipelineStagePayloadType
from savant_rs.primitives import EndOfStream, Shutdown, VideoFrame
from savant_rs.utils import TelemetrySpan
from savant_rs.utils.serialization import Message, save_message_to_bytes

from savant.client.frame_source import FrameSource
from savant.client.log_provider import LogProvider
from savant.client.runner import LogResult
from savant.client.runner.healthcheck import HealthCheck
from savant.healthcheck.status import PipelineStatus
from savant.utils.logging import get_logger
from savant.utils.zeromq import (
    Defaults,
    SenderSocketTypes,
    async_receive_response,
    parse_zmq_socket_uri,
    receive_response,
)

logger = get_logger(__name__)

Frame = Union[FrameSource, Tuple[VideoFrame, bytes]]
FrameAndEos = Union[Frame, EndOfStream]


@dataclass
class SourceResult(LogResult):
    """Result of sending a message to ZeroMQ socket."""

    source_id: str
    """Source ID."""
    pts: Optional[int]
    """PTS of the frame."""
    status: str
    """Status of sending the message."""


class SourceRunner:
    """Sends messages to ZeroMQ socket."""

    def __init__(
        self,
        socket: str,
        log_provider: Optional[LogProvider],
        retries: int,
        module_health_check_url: Optional[str],
        module_health_check_timeout: float,
        module_health_check_interval: float,
        telemetry_enabled: bool,
        send_hwm: int = Defaults.SEND_HWM,
        receive_timeout: int = Defaults.SENDER_RECEIVE_TIMEOUT,
    ):
        self._socket = socket
        self._log_provider = log_provider
        self._retries = retries
        self._telemetry_enabled = telemetry_enabled
        self._send_hwm = send_hwm
        self._receive_timeout = receive_timeout
        self._health_check = (
            HealthCheck(
                url=module_health_check_url,
                interval=module_health_check_interval,
                timeout=module_health_check_timeout,
                ready_statuses=[PipelineStatus.RUNNING],
            )
            if module_health_check_url is not None
            else None
        )
        self._socket_type, self._bind, self._socket = parse_zmq_socket_uri(
            uri=socket,
            socket_type_enum=SenderSocketTypes,
            socket_type_name=None,
            bind=None,
        )

        self._last_send_time = 0
        self._wait_response = self._socket_type == SenderSocketTypes.REQ
        self._zmq_context = self._create_zmq_ctx()
        self._sender = self._zmq_context.socket(self._socket_type.value)
        self._sender.setsockopt(zmq.SNDHWM, self._send_hwm)
        self._sender.setsockopt(zmq.RCVTIMEO, self._receive_timeout)
        if self._bind:
            self._sender.bind(self._socket)
        else:
            self._sender.connect(self._socket)

        self._pipeline_stage_name = 'savant-client'
        self._pipeline = VideoPipeline(
            'savant-client',
            [(self._pipeline_stage_name, VideoPipelineStagePayloadType.Frame)],
        )
        if self._telemetry_enabled:
            self._pipeline.sampling_period = 1

    def __call__(self, source: Frame, send_eos: bool = True) -> SourceResult:
        """Send a single frame to ZeroMQ socket.

        :param source: Source of the frame to send. Can be an instance
            of FrameSource or a tuple of VideoFrame and content.
        :param send_eos: Whether to send EOS after sending the frame.
        :return: Result of sending the frame.
        """

        return self.send(source, send_eos)

    def send(self, source: Frame, send_eos: bool = True) -> SourceResult:
        """Send a single frame to ZeroMQ socket.

        :param source: Source of the frame to send. Can be an instance
            of FrameSource or a tuple of VideoFrame and content.
        :param send_eos: Whether to send EOS after sending the frame.
        :return: Result of sending the frame.
        """

        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        zmq_topic, serialized_message, content, result = self._prepare_video_frame(
            source
        )
        logger.debug(
            'Sending video frame %s/%s.',
            result.source_id,
            result.pts,
        )
        self._send_zmq_message([zmq_topic, serialized_message, content])
        logger.debug('Sent video frame %s/%s.', result.source_id, result.pts)
        if send_eos:
            self.send_eos(result.source_id)
        result.status = 'ok'

        return result

    def send_iter(
        self,
        sources: Iterable[FrameAndEos],
        send_eos: bool = True,
    ) -> Iterable[SourceResult]:
        """Send multiple frames to ZeroMQ socket.

        :param sources: Sources of the frames to send.
        :param send_eos: Whether to send EOS after sending the frames.
        :return: Results of sending the frames.
        """

        source_ids = set()
        for source in sources:
            if isinstance(source, EndOfStream):
                self.send_eos(source.source_id)
                source_ids.remove(source.source_id)
                continue

            result = self.send(source, send_eos=False)
            source_ids.add(result.source_id)
            yield result
        if send_eos:
            for source_id in source_ids:
                self.send_eos(source_id)

    def send_eos(self, source_id: str) -> SourceResult:
        """Send EOS for a source to ZeroMQ socket.

        :param source_id: Source ID.
        :return: Result of sending EOS.
        """

        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        zmq_topic, serialized_message, result = self._prepare_eos(source_id)
        self._send_zmq_message([zmq_topic, serialized_message])
        logger.debug('Sent EOS for source %s.', source_id)
        result.status = 'ok'

        return result

    def send_shutdown(self, source_id: str, auth: str) -> SourceResult:
        """Send Shutdown message for a source to ZeroMQ socket.

        :param source_id: Source ID.
        :param auth: Authentication key.
        :return: Result of sending Shutdown.
        """

        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        zmq_topic, serialized_message, result = self._prepare_shutdown(source_id, auth)
        self._send_zmq_message([zmq_topic, serialized_message])
        logger.debug('Sent Shutdown message for source %s.', source_id)
        result.status = 'ok'

        return result

    def _send_zmq_message(self, message: List[bytes]):
        for retries_left in reversed(range(self._retries)):
            try:
                self._sender.send_multipart(message)
                break
            except Exception:
                if retries_left == 0:
                    raise
                logger.error(
                    'Failed to send message to socket %s. %s retries left.',
                    self._socket,
                    retries_left,
                    exc_info=True,
                )
        if self._wait_response:
            receive_response(self._sender, self._retries)

    def _create_zmq_ctx(self):
        return zmq.Context()

    def _prepare_video_frame(self, source: Frame):
        if isinstance(source, FrameSource):
            logger.debug('Sending video frame from source %s.', source)
            video_frame, content = source.build_frame()
        else:
            video_frame, content = source
            logger.debug('Sending video frame from source %s.', video_frame.source_id)
        frame_id = self._pipeline.add_frame(self._pipeline_stage_name, video_frame)
        zmq_topic = f'{video_frame.source_id}/'.encode()
        message = Message.video_frame(video_frame)
        if self._telemetry_enabled:
            span: TelemetrySpan = self._pipeline.delete(frame_id)[frame_id]
            message.span_context = span.propagate()
            trace_id = span.trace_id()
            del span
        else:
            trace_id = None
        serialized_message = save_message_to_bytes(message)

        return (
            zmq_topic,
            serialized_message,
            content,
            SourceResult(
                source_id=video_frame.source_id,
                pts=video_frame.pts,
                status='',
                trace_id=trace_id,
                log_provider=self._log_provider,
            ),
        )

    def _prepare_eos(self, source_id: str):
        logger.debug('Sending EOS for source %s.', source_id)
        zmq_topic = f'{source_id}/'.encode()
        message = Message.end_of_stream(EndOfStream(source_id))
        serialized_message = save_message_to_bytes(message)

        return (
            zmq_topic,
            serialized_message,
            SourceResult(
                source_id=source_id,
                pts=None,
                status='',
                trace_id=None,
                log_provider=self._log_provider,
            ),
        )

    def _prepare_shutdown(self, source_id: str, auth: str):
        logger.debug('Sending Shutdown message for source %s.', source_id)
        zmq_topic = f'{source_id}/'.encode()
        message = Message.shutdown(Shutdown(auth))
        serialized_message = save_message_to_bytes(message)

        return (
            zmq_topic,
            serialized_message,
            SourceResult(
                source_id=source_id,
                pts=None,
                status='',
                trace_id=None,
                log_provider=self._log_provider,
            ),
        )


class AsyncSourceRunner(SourceRunner):
    """Sends messages to ZeroMQ socket asynchronously."""

    async def __call__(self, source: Frame, send_eos: bool = True) -> SourceResult:
        return await self.send(source, send_eos)

    async def send(self, source: Frame, send_eos: bool = True) -> SourceResult:
        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        zmq_topic, serialized_message, content, result = self._prepare_video_frame(
            source
        )
        logger.debug(
            'Sending video frame %s/%s.',
            result.source_id,
            result.pts,
        )
        await self._send_zmq_message([zmq_topic, serialized_message, content])
        logger.debug('Sent video frame %s/%s.', result.source_id, result.pts)
        if send_eos:
            await self.send_eos(result.source_id)

        result.status = 'ok'

        return result

    async def send_iter(
        self,
        sources: Union[Iterable[FrameAndEos], AsyncIterable[FrameAndEos]],
        send_eos: bool = True,
    ) -> AsyncIterable[SourceResult]:
        source_ids = set()
        if isinstance(sources, AsyncIterable):
            async for source in sources:
                yield await self._send_iter_item(source, source_ids)
        else:
            for source in sources:
                yield await self._send_iter_item(source, source_ids)
        if send_eos:
            for source_id in source_ids:
                await self.send_eos(source_id)

    async def send_eos(self, source_id: str) -> SourceResult:
        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        zmq_topic, serialized_message, result = self._prepare_eos(source_id)
        await self._send_zmq_message([zmq_topic, serialized_message])
        logger.debug('Sent EOS for source %s.', source_id)
        result.status = 'ok'

        return result

    async def send_shutdown(self, source_id: str, auth: str) -> SourceResult:
        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        zmq_topic, serialized_message, result = self._prepare_shutdown(source_id, auth)
        await self._send_zmq_message([zmq_topic, serialized_message])
        logger.debug('Sent Shutdown message for source %s.', source_id)
        result.status = 'ok'

        return result

    async def _send_zmq_message(self, message: List[bytes]):
        for retries_left in reversed(range(self._retries)):
            try:
                await self._sender.send_multipart(message)
                break
            except Exception:
                if retries_left == 0:
                    raise
                logger.error(
                    'Failed to send message to socket %s. %s retries left.',
                    self._socket,
                    retries_left,
                    exc_info=True,
                )
        if self._wait_response:
            await async_receive_response(self._sender, self._retries)

    def _create_zmq_ctx(self):
        return zmq.asyncio.Context()

    async def _send_iter_item(self, source: FrameAndEos, source_ids: Set[str]):
        if isinstance(source, EndOfStream):
            await self.send_eos(source.source_id)
            source_ids.remove(source.source_id)
            return

        result = await self.send(source, send_eos=False)
        source_ids.add(result.source_id)
        return result
