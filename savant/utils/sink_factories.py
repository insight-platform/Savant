"""Sink factories."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Union

import zmq
from savant_rs.primitives import EndOfStream, VideoFrame, VideoFrameContent
from savant_rs.utils import PropagatedContext
from savant_rs.utils.serialization import Message, save_message_to_bytes

from savant.api.enums import ExternalFrameType
from savant.api.parser import convert_ts
from savant.base.pyfunc import PyFunc
from savant.config.schema import SinkElement
from savant.utils.logging import get_logger
from savant.utils.registry import Registry
from savant.utils.zeromq import (
    Defaults,
    SenderSocketTypes,
    create_ipc_socket_dirs,
    ipc_socket_chmod,
    parse_zmq_socket_uri,
    receive_response,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class SinkMessage:
    """Sink message."""

    @property
    def source_id(self) -> str:
        pass

    def _replace(self, **kwargs):
        return replace(self, **kwargs)


@dataclass(frozen=True)
class SinkVideoFrame(SinkMessage):
    """Message for VideoFrame message schema."""

    video_frame: VideoFrame
    frame: Optional[bytes]
    span_context: Optional[PropagatedContext] = None

    @property
    def source_id(self) -> str:
        return self.video_frame.source_id


@dataclass(frozen=True)
class SinkEndOfStream(SinkMessage):
    """Message for EndOfStream message schema."""

    eos: EndOfStream

    @property
    def source_id(self) -> str:
        return self.eos.source_id


SinkCallable = Callable[[SinkMessage, Dict[str, Any]], None]


class SinkFactory(ABC):
    """SinkFactory interface."""

    def __init__(self, sink_name: str, egress_pyfunc: PyFunc) -> None:
        self.name = sink_name
        self.egress_pyfunc = egress_pyfunc
        try:
            self.egress_pyfunc.load_user_code()
        except Exception as exc:
            logger.warning(
                'Error in sink "%s" loading egress filter %s: %s. '
                'Using noop placeholder.',
                self.name,
                self.egress_pyfunc,
                exc,
            )
            self.egress_pyfunc = lambda x: True

    def video_frame_filter(self, video_frame: VideoFrame, frame_pts: int) -> bool:
        """Wrapper for egress frame filter."""
        try:
            ret = self.egress_pyfunc(video_frame)
        except Exception as exc:
            logger.warning(
                'Frame of source "%s" with PTS %s got error in sink "%s" '
                'egress filter %s call: %s. Allowing frame to pass.',
                video_frame.source_id,
                frame_pts,
                self.name,
                self.egress_pyfunc,
                exc,
            )
            return True

        if ret:
            logger.debug(
                'Frame of source "%s" with PTS %s passed "%s" sink egress filter.',
                video_frame.source_id,
                frame_pts,
                self.name,
            )
        else:
            logger.debug(
                'Frame of source "%s" with PTS %s didnt pass "%s" sink egress filter,'
                'skipping it.',
                video_frame.source_id,
                frame_pts,
                self.name,
            )
        return ret

    @abstractmethod
    def get_sink(self) -> SinkCallable:
        """Sink factory method."""


class MultiSinkFactory(SinkFactory):
    """Multiple sink combination, message is sent to each one.

    :param factories: sink factories.
    """

    def __init__(self, *factories: SinkFactory):
        self.factories = factories

    def get_sink(self) -> SinkCallable:
        sinks = [x.get_sink() for x in self.factories]

        def send_message(
            msg: SinkMessage,
            **kwargs,
        ):
            for sink in sinks:
                sink(msg, **kwargs)

        return send_message


SINK_REGISTRY = Registry('sink')


@SINK_REGISTRY.register('zeromq_sink')
class ZeroMQSinkFactory(SinkFactory):
    """ZeroMQ sink factory.

    :param socket: zeromq socket.
    :param socket_type: zeromq socket type.
    :param bind: indicates whether the client should bind or connect to zeromq socket.
    :param send_hwm: high watermark for outbound messages.
    """

    def __init__(
        self,
        sink_name: str,
        egress_pyfunc: PyFunc,
        socket: str,
        socket_type: str = SenderSocketTypes.PUB.name,
        bind: bool = True,
        send_hwm: int = Defaults.SEND_HWM,
        receive_timeout: int = Defaults.SENDER_RECEIVE_TIMEOUT,
        req_receive_retries: int = Defaults.REQ_RECEIVE_RETRIES,
        set_ipc_socket_permissions: bool = True,
    ):
        super().__init__(sink_name, egress_pyfunc)
        logger.debug(
            'Initializing ZMQ sink: socket %s, type %s, bind %s.',
            socket,
            socket_type,
            bind,
        )

        self.receive_timeout = receive_timeout
        self.req_receive_retries = req_receive_retries
        self.set_ipc_socket_permissions = set_ipc_socket_permissions
        # might raise exceptions
        # will be handled in savant.entrypoint
        self.socket_type, self.bind, self.socket = parse_zmq_socket_uri(
            uri=socket,
            socket_type_name=socket_type,
            socket_type_enum=SenderSocketTypes,
            bind=bind,
        )

        self.send_hwm = send_hwm
        self.wait_response = self.socket_type == SenderSocketTypes.REQ

    def get_sink(self) -> SinkCallable:
        context = zmq.Context()
        output_zmq_socket = context.socket(self.socket_type.value)
        output_zmq_socket.setsockopt(zmq.SNDHWM, self.send_hwm)
        output_zmq_socket.setsockopt(zmq.RCVTIMEO, self.receive_timeout)

        create_ipc_socket_dirs(self.socket)

        if self.bind:
            output_zmq_socket.bind(self.socket)
        else:
            output_zmq_socket.connect(self.socket)
        if self.set_ipc_socket_permissions and self.bind:
            ipc_socket_chmod(self.socket)

        def send_message(
            msg: SinkMessage,
            **kwargs,
        ):
            zmq_topic = f'{msg.source_id}/'.encode()
            zmq_message = [zmq_topic]

            if isinstance(msg, SinkVideoFrame):
                frame_pts = convert_ts(msg.video_frame.pts, msg.video_frame.time_base)
                if self.video_frame_filter(msg.video_frame, frame_pts):
                    logger.debug(
                        'Sending frame of source "%s" with PTS %s to ZeroMQ sink.',
                        msg.source_id,
                        frame_pts,
                    )

                    if msg.frame:
                        logger.debug(
                            'Size of frame of source "%s" with PTS %s is %s bytes.',
                            msg.source_id,
                            frame_pts,
                            len(msg.frame),
                        )
                        msg.video_frame.content = VideoFrameContent.external(
                            ExternalFrameType.ZEROMQ.value, None
                        )
                    else:
                        logger.debug(
                            'Frame of source "%s" with PTS %s is empty.',
                            msg.source_id,
                            frame_pts,
                        )
                        msg.video_frame.content = VideoFrameContent.none()

                    message = Message.video_frame(msg.video_frame)
                    if msg.span_context is not None:
                        message.span_context = msg.span_context
                    zmq_message.append(save_message_to_bytes(message))
                    if msg.frame:
                        zmq_message.append(msg.frame)
            elif isinstance(msg, SinkEndOfStream):
                logger.debug(
                    'Sending EOS of source "%s" to ZeroMQ sink.', msg.source_id
                )
                message = Message.end_of_stream(msg.eos)
                zmq_message.append(save_message_to_bytes(message))
            else:
                logger.warning('Unknown message type %s.', type(msg))
                return
            output_zmq_socket.send_multipart(zmq_message)
            if self.wait_response:
                resp = receive_response(output_zmq_socket, self.req_receive_retries)
                logger.debug(
                    'Received %s bytes from socket %s.', len(resp), self.socket
                )

        return send_message


@SINK_REGISTRY.register('console_sink')
class ConsoleSinkFactory(SinkFactory):
    """Just output messages to STDOUT."""

    def __init__(self, sink_name: str, egress_pyfunc: PyFunc, json_mode: bool = False):
        super().__init__(sink_name, egress_pyfunc)
        self.json_mode = json_mode

    def get_sink(self) -> SinkCallable:
        if self.json_mode:

            def format_meta(video_frame: VideoFrame) -> str:
                return video_frame.json_pretty

        else:

            def format_meta(video_frame: VideoFrame) -> str:
                return str(video_frame)

        def send_message(
            msg: SinkMessage,
            **kwargs,
        ):

            if isinstance(msg, SinkVideoFrame):
                frame_pts = convert_ts(msg.video_frame.pts, msg.video_frame.time_base)
                if self.video_frame_filter(msg.video_frame, frame_pts):
                    message = f'Frame shape(WxH): {msg.video_frame.width}x{msg.video_frame.height}'
                    if msg.video_frame.codec is not None:
                        message += f', codec: {msg.video_frame.codec}'
                    if msg.frame is not None:
                        message += f', size (bytes): {len(msg.frame)}'
                    message += f'.\nMeta: {format_meta(msg.video_frame)}.\n'
                    print(message)

            elif isinstance(msg, SinkEndOfStream):
                message = f'End of stream {msg.source_id}.\n'
                print(message)

        return send_message


class JsonWriter:
    """Output messages to json file.

    :param file_path: path to output file.
    """

    def __init__(self, file_path: str):
        self._json_file = open(file_path, 'w', encoding='utf8')
        self._json_file.write('[')
        self._first = True

    def write(self, message: str):
        """Write message to a file in json format."""
        if not self._first:
            self._json_file.write(',')
        self._json_file.write(message)
        self._first = False

    def __del__(self):
        self._json_file.write(']')
        self._json_file.close()


@SINK_REGISTRY.register('file_sink')
class FileSinkFactory(SinkFactory):
    """Sink that outputs messages to json file.

    :param file_path: path to output file.
    """

    def __init__(self, sink_name: str, egress_pyfunc: PyFunc, file_path: str):
        super().__init__(sink_name, egress_pyfunc)
        self._json_writer = JsonWriter(file_path)

    def get_sink(self) -> SinkCallable:
        def send_message(
            msg: SinkMessage,
            **kwargs,
        ):
            if isinstance(msg, SinkVideoFrame):
                frame_pts = convert_ts(msg.video_frame.pts, msg.video_frame.time_base)
                if self.video_frame_filter(msg.video_frame, frame_pts):
                    self._json_writer.write(msg.video_frame.json)

        return send_message


@SINK_REGISTRY.register('devnull_sink')
class DevNullSinkFactory(SinkFactory):
    """Do nothing, similar to `> /dev/null`.

    We must always specify sink. Use this sink to check that the
    pipeline is working or to measure pipeline performance (FPS).
    """

    def get_sink(self) -> SinkCallable:
        def send_message(
            msg: SinkMessage,
            **kwargs,
        ):
            pass

        return send_message


def sink_factory(sink: Union[SinkElement, List[SinkElement]]) -> SinkCallable:
    """Init sink from config."""
    if isinstance(sink, SinkElement):
        return SINK_REGISTRY.get(sink.element.lower())(
            sink.full_name, sink.egress_frame_filter, **sink.properties
        ).get_sink()

    sink_factories = []
    for _sink in sink:
        sink_factories.append(
            SINK_REGISTRY.get(_sink.element.lower())(
                _sink.full_name, _sink.egress_frame_filter, **_sink.properties
            )
        )
    return MultiSinkFactory(*sink_factories).get_sink()
