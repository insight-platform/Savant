import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from savant_rs.primitives import EndOfStream, VideoFrame, VideoFrameContent
from savant_rs.utils.serialization import Message, load_message_from_bytes

from savant.client.log_provider import LogProvider
from savant.client.runner import LogResult
from savant.utils.zeromq import Defaults, ZeroMQSource

logger = logging.getLogger(__name__)


@dataclass
class SinkResult(LogResult):
    frame_meta: Optional[VideoFrame]
    frame_content: Optional[bytes] = field(repr=False)
    eos: Optional[EndOfStream]


class SinkRunner:
    def __init__(
        self,
        socket: str,
        log_provider: Optional[LogProvider] = None,
        receive_timeout: int = Defaults.RECEIVE_TIMEOUT,
        receive_hwm: int = Defaults.RECEIVE_HWM,
        idle_timeout: Optional[int] = None,
    ):
        self._log_provider = log_provider
        self._idle_timeout = idle_timeout if idle_timeout is not None else 10**6
        self._source = ZeroMQSource(
            socket=socket,
            receive_timeout=receive_timeout,
            receive_hwm=receive_hwm,
            set_ipc_socket_permissions=False,
        )
        self._source.start()

    def __next__(self) -> SinkResult:
        wait_until = time.time() + self._idle_timeout
        result = None
        while result is None:
            result = self.receive_next_message()
            if result is None and time.time() > wait_until:
                raise StopIteration()

        return result

    def __iter__(self):
        return self

    def receive_next_message(self) -> Optional[SinkResult]:
        message_parts = self._source.next_message()
        if message_parts is None:
            return None

        message: Message = load_message_from_bytes(message_parts[0])
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
            if len(message_parts) > 1:
                content = message_parts[1]
            else:
                content = None
                if video_frame.content.internal():
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
