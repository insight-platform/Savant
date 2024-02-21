import asyncio
from dataclasses import dataclass
from typing import AsyncIterable, Iterable, Optional, Set, Tuple, Union

from savant_rs.pipeline2 import (
    VideoPipeline,
    VideoPipelineConfiguration,
    VideoPipelineStagePayloadType,
)
from savant_rs.primitives import EndOfStream, Shutdown, VideoFrame
from savant_rs.utils import TelemetrySpan
from savant_rs.utils.serialization import Message, clear_source_seq_id
from savant_rs.zmq import (
    BlockingWriter,
    NonBlockingWriter,
    WriterConfig,
    WriterConfigBuilder,
)

from savant.client.frame_source import FrameSource
from savant.client.log_provider import LogProvider
from savant.client.runner import LogResult
from savant.client.runner.healthcheck import HealthCheck
from savant.healthcheck.status import ModuleStatus
from savant.utils.logging import get_logger
from savant.utils.zeromq import Defaults

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

    _writer: BlockingWriter

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
                ready_statuses=[ModuleStatus.RUNNING],
            )
            if module_health_check_url is not None
            else None
        )

        config_builder = WriterConfigBuilder(socket)
        config_builder.with_receive_timeout(receive_timeout)
        config_builder.with_send_hwm(send_hwm)
        config = config_builder.build()

        self._last_send_time = 0
        self._writer = self._build_zeromq_writer(config)

        self._pipeline_stage_name = 'savant-client'
        self._pipeline = VideoPipeline(
            'savant-client',
            [(self._pipeline_stage_name, VideoPipelineStagePayloadType.Frame)],
            VideoPipelineConfiguration(),
        )
        if self._telemetry_enabled:
            self._pipeline.sampling_period = 1
        self._writer.start()

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

        zmq_topic, message, content, result = self._prepare_video_frame(source)
        logger.debug(
            'Sending video frame %s/%s.',
            result.source_id,
            result.pts,
        )
        self._send_zmq_message(zmq_topic, message, content)
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

        zmq_topic, message, result = self._prepare_eos(source_id)
        self._send_zmq_message(zmq_topic, message)
        logger.debug('Sent EOS for source %s.', source_id)
        result.status = 'ok'
        clear_source_seq_id(source_id)
        return result

    def send_shutdown(self, source_id: str, auth: str) -> SourceResult:
        """Send Shutdown message for a source to ZeroMQ socket.

        :param source_id: Source ID.
        :param auth: Authentication key.
        :return: Result of sending Shutdown.
        """

        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        zmq_topic, message, result = self._prepare_shutdown(source_id, auth)
        self._send_zmq_message(zmq_topic, message)
        logger.debug('Sent Shutdown message for source %s.', source_id)
        result.status = 'ok'

        return result

    def _send_zmq_message(self, topic: str, message: Message, content: bytes = b''):
        self._writer.send_message(topic, message, content)

    def _build_zeromq_writer(self, config: WriterConfig):
        return BlockingWriter(config)

    def _prepare_video_frame(self, source: Frame):
        if isinstance(source, FrameSource):
            logger.debug('Sending video frame from source %s.', source)
            video_frame, content = source.build_frame()
        else:
            video_frame, content = source
            logger.debug('Sending video frame from source %s.', video_frame.source_id)
        frame_id = self._pipeline.add_frame(self._pipeline_stage_name, video_frame)
        message = video_frame.to_message()
        if self._telemetry_enabled:
            span: TelemetrySpan = self._pipeline.delete(frame_id)[frame_id]
            message.span_context = span.propagate()
            trace_id = span.trace_id()
            del span
        else:
            trace_id = None

        return (
            video_frame.source_id,
            message,
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
        message = EndOfStream(source_id).to_message()

        return (
            source_id,
            message,
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
        message = Shutdown(auth).to_message()

        return (
            source_id,
            message,
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

    _writer: NonBlockingWriter

    async def __call__(self, source: Frame, send_eos: bool = True) -> SourceResult:
        return await self.send(source, send_eos)

    async def send(self, source: Frame, send_eos: bool = True) -> SourceResult:
        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        zmq_topic, message, content, result = self._prepare_video_frame(source)
        logger.debug(
            'Sending video frame %s/%s.',
            result.source_id,
            result.pts,
        )
        await self._send_zmq_message(zmq_topic, message, content)
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

        zmq_topic, message, result = self._prepare_eos(source_id)
        await self._send_zmq_message(zmq_topic, message)
        logger.debug('Sent EOS for source %s.', source_id)
        result.status = 'ok'
        clear_source_seq_id(source_id)
        return result

    async def send_shutdown(self, source_id: str, auth: str) -> SourceResult:
        if self._health_check is not None:
            self._health_check.wait_module_is_ready()

        zmq_topic, message, result = self._prepare_shutdown(source_id, auth)
        await self._send_zmq_message(zmq_topic, message)
        logger.debug('Sent Shutdown message for source %s.', source_id)
        result.status = 'ok'

        return result

    async def _send_zmq_message(
        self, topic: str, message: Message, content: bytes = b''
    ):
        while not self._writer.has_capacity():
            await asyncio.sleep(0.01)  # TODO: make configurable
        await asyncio.get_running_loop().run_in_executor(
            None, self._writer.send_message, topic, message, content
        )

    def _build_zeromq_writer(self, config: WriterConfig):
        return NonBlockingWriter(config, 10)  # TODO: make configurable

    async def _send_iter_item(self, source: FrameAndEos, source_ids: Set[str]):
        if isinstance(source, EndOfStream):
            await self.send_eos(source.source_id)
            source_ids.remove(source.source_id)
            return

        result = await self.send(source, send_eos=False)
        source_ids.add(result.source_id)
        return result
