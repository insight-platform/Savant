from dataclasses import dataclass, field
from typing import Optional

from savant_rs.primitives import EndOfStream, VideoFrame, VideoFrameContent
from savant_rs.utils.serialization import Message, load_message_from_bytes

from savant.source_sink_framework.log_provider import LogProvider
from savant.utils.zeromq import Defaults, ZeroMQSource


@dataclass
class SinkResult:
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
    ):
        self._log_provider = log_provider
        self._source = ZeroMQSource(
            socket=socket,
            receive_timeout=receive_timeout,
            receive_hwm=receive_hwm,
            set_ipc_socket_permissions=False,
        )
        self._source.start()

    def __next__(self) -> SinkResult:
        result = None
        while result is None:
            result = self.receive_next_message()

        return result

    def __iter__(self):
        return self

    def receive_next_message(self) -> Optional[SinkResult]:
        message_parts = self._source.next_message()
        if message_parts is None:
            return None

        message: Message = load_message_from_bytes(message_parts[0])
        if message.is_video_frame():
            video_frame: VideoFrame = message.as_video_frame()
            if len(message_parts) > 1:
                content = message_parts[1]
            else:
                content = None
                if video_frame.content.internal():
                    content = video_frame.content.get_data_as_bytes()
                    video_frame.content = VideoFrameContent.none()
            return SinkResult(video_frame, content, None)

        if message.is_end_of_stream():
            return SinkResult(None, None, message.as_end_of_stream())

        raise Exception('Unknown message type')
