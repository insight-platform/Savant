import logging
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import zmq
from savant_rs.primitives import EndOfStream
from savant_rs.utils.serialization import Message, save_message_to_bytes

from savant.client.frame_source import FrameSource
from savant.client.log_provider import LogProvider
from savant.utils.zeromq import (
    Defaults,
    SenderSocketTypes,
    parse_zmq_socket_uri,
    receive_response,
)

logger = logging.getLogger(__name__)


@dataclass
class SourceResult:
    source_id: str
    status: str


class SourceRunner:
    def __init__(
        self,
        timeout: float,
        socket: str,
        log_provider: Optional[LogProvider] = None,
        send_hwm: int = Defaults.SEND_HWM,
        receive_timeout: int = Defaults.SENDER_RECEIVE_TIMEOUT,
        req_receive_retries: int = Defaults.REQ_RECEIVE_RETRIES,
    ):
        self._timeout = timeout
        self._socket = socket
        self._log_provider = log_provider
        self._send_hwm = send_hwm
        self._receive_timeout = receive_timeout
        self._req_receive_retries = req_receive_retries
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

    def __call__(self, source: FrameSource) -> SourceResult:
        return self.send(source)

    def send(self, source: FrameSource, send_eos: bool = True) -> SourceResult:
        logger.debug('Sending video frame from source %s.', source)
        video_frame, content = source.build_frame()
        zmq_topic = f'{video_frame.source_id}/'.encode()
        message = Message.video_frame(video_frame)
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

        return SourceResult(video_frame.source_id, 'ok')

    def send_iter(
        self,
        sources: Iterable[FrameSource],
        send_eos: bool = True,
    ) -> Iterable[SourceResult]:
        source_ids = set()
        for source in sources:
            result = self.send(source, send_eos=False)
            source_ids.add(result.source_id)
            yield result
        if send_eos:
            for source_id in source_ids:
                self.send_eos(source_id)

    def send_eos(self, source_id: str):
        logger.debug('Sending EOS for source %s.', source_id)
        zmq_topic = f'{source_id}/'.encode()
        message = Message.end_of_stream(EndOfStream(source_id))
        serialized_message = save_message_to_bytes(message)
        self._send_zmq_message([zmq_topic, serialized_message])
        logger.debug('Sent EOS for source %s.', source_id)

        return SourceResult(source_id, 'ok')

    def _send_zmq_message(self, message: List[bytes]):
        if self._timeout:
            timeout = self._last_send_time + self._timeout - time.time()
            if timeout > 0:
                time.sleep(timeout)
        self._last_send_time = time.time()
        self._sender.send_multipart(message)
        if self._wait_response:
            receive_response(self._sender, self._req_receive_retries)

    def _wait_timeout(self):
        if self._timeout:
            timeout = self._last_send_time + self._timeout - time.time()
            if timeout > 0:
                time.sleep(timeout)
        self._last_send_time = time.time()
