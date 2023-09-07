from dataclasses import dataclass
from typing import Iterable, List, Optional

import zmq
from savant_rs.pipeline2 import VideoPipeline, VideoPipelineStagePayloadType
from savant_rs.primitives import EndOfStream, Shutdown
from savant_rs.utils import TelemetrySpan
from savant_rs.utils.serialization import Message, save_message_to_bytes

from savant.client.frame_source import FrameSource
from savant.client.log_provider import LogProvider
from savant.client.runner import LogResult
from savant.utils.logging import get_logger
from savant.utils.zeromq import (
    Defaults,
    SenderSocketTypes,
    parse_zmq_socket_uri,
    receive_response,
)

logger = get_logger(__name__)


@dataclass
class SourceResult(LogResult):
    """Result of sending a message to ZeroMQ socket."""

    source_id: str
    """Source ID."""
    status: str
    """Status of sending the message."""


class SourceRunner:
    """Sends messages to ZeroMQ socket."""

    def __init__(
        self,
        socket: str,
        log_provider: Optional[LogProvider] = None,
        retries: int = Defaults.REQ_RECEIVE_RETRIES,
        send_hwm: int = Defaults.SEND_HWM,
        receive_timeout: int = Defaults.SENDER_RECEIVE_TIMEOUT,
    ):
        self._socket = socket
        self._log_provider = log_provider
        self._retries = retries
        self._send_hwm = send_hwm
        self._receive_timeout = receive_timeout
        self._socket_type, self._bind, self._socket = parse_zmq_socket_uri(
            uri=socket,
            socket_type_enum=SenderSocketTypes,
            socket_type_name=None,
            bind=None,
        )

        self._last_send_time = 0
        self._wait_response = self._socket_type == SenderSocketTypes.REQ
        self._zmq_context: zmq.Context = zmq.Context()
        self._sender: zmq.Socket = self._zmq_context.socket(self._socket_type.value)
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
        self._pipeline.sampling_period = 1

    def __call__(self, source: FrameSource, send_eos: bool = True) -> SourceResult:
        """Send a single frame to ZeroMQ socket.

        :param source: Source of the frame to send.
        :param send_eos: Whether to send EOS after sending the frame.
        :return: Result of sending the frame.
        """

        return self.send(source, send_eos)

    def send(self, source: FrameSource, send_eos: bool = True) -> SourceResult:
        """Send a single frame to ZeroMQ socket.

        :param source: Source of the frame to send.
        :param send_eos: Whether to send EOS after sending the frame.
        :return: Result of sending the frame.
        """

        logger.debug('Sending video frame from source %s.', source)
        video_frame, content = source.build_frame()
        frame_id = self._pipeline.add_frame(self._pipeline_stage_name, video_frame)
        zmq_topic = f'{video_frame.source_id}/'.encode()
        message = Message.video_frame(video_frame)
        span: TelemetrySpan = self._pipeline.delete(frame_id)[frame_id]
        message.span_context = span.propagate()
        trace_id = span.trace_id()
        del span
        serialized_message = save_message_to_bytes(message)
        logger.debug(
            'Sending video frame %s/%s.',
            video_frame.source_id,
            video_frame.pts,
        )
        self._send_zmq_message([zmq_topic, serialized_message, content])
        logger.debug('Sent video frame %s/%s.', video_frame.source_id, video_frame.pts)
        if send_eos:
            self.send_eos(video_frame.source_id)

        return SourceResult(
            source_id=video_frame.source_id,
            status='ok',
            trace_id=trace_id,
            log_provider=self._log_provider,
        )

    def send_iter(
        self,
        sources: Iterable[FrameSource],
        send_eos: bool = True,
    ) -> Iterable[SourceResult]:
        """Send multiple frames to ZeroMQ socket.

        :param sources: Sources of the frames to send.
        :param send_eos: Whether to send EOS after sending the frames.
        :return: Results of sending the frames.
        """

        source_ids = set()
        for source in sources:
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

        logger.debug('Sending EOS for source %s.', source_id)
        zmq_topic = f'{source_id}/'.encode()
        message = Message.end_of_stream(EndOfStream(source_id))
        serialized_message = save_message_to_bytes(message)
        self._send_zmq_message([zmq_topic, serialized_message])
        logger.debug('Sent EOS for source %s.', source_id)

        return SourceResult(
            source_id=source_id,
            status='ok',
            trace_id=None,
            log_provider=self._log_provider,
        )

    def send_shutdown(self, source_id: str, auth: str) -> SourceResult:
        """Send Shutdown message for a source to ZeroMQ socket.

        :param source_id: Source ID.
        :param auth: Authentication key.
        :return: Result of sending Shutdown.
        """

        logger.debug('Sending Shutdown message for source %s.', source_id)
        zmq_topic = f'{source_id}/'.encode()
        message = Message.shutdown(Shutdown(auth))
        serialized_message = save_message_to_bytes(message)
        self._send_zmq_message([zmq_topic, serialized_message])
        logger.debug('Sent Shutdown message for source %s.', source_id)

        return SourceResult(
            source_id=source_id,
            status='ok',
            trace_id=None,
            log_provider=self._log_provider,
        )

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
