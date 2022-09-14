"""Sink factories."""
import json
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union
import logging

import zmq

from savant.api import ENCODING_REGISTRY, serialize
from savant.config.schema import PipelineElement
from savant.gstreamer.codecs import CodecInfo
from savant.gstreamer.metadata import GstFrameMeta
from savant.utils.registry import Registry
from savant.utils.zeromq import SenderSocketTypes

logger = logging.getLogger(__name__)


class SinkMessage:
    """Sink message."""

    source_id: str


class SinkVideoFrame(SinkMessage, NamedTuple):
    """Message for VideoFrame message schema."""

    source_id: str
    frame_meta: GstFrameMeta
    frame_width: int
    frame_height: int
    frame: Optional[bytes]
    frame_codec: Optional[CodecInfo]
    keyframe: bool


class SinkEndOfStream(SinkMessage, NamedTuple):
    """Message for EndOfStream message schema."""

    source_id: str


SinkCallable = Callable[[SinkMessage, Dict[str, Any]], None]


class SinkFactory(ABC):
    """SinkFactory interface."""

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
    """

    def __init__(
        self,
        socket: str,
        socket_type: str = SenderSocketTypes.PUB.name,
        bind: bool = True,
    ):
        self.socket = socket
        try:
            self.socket_type = SenderSocketTypes[socket_type]
        except KeyError as exc:
            raise AttributeError(f'Incorrect socket type: {socket_type}.') from exc
        self.bind = bind
        logger.debug(
            'init ZMQ sink, socket %s, type %s, bind %s.', socket, socket_type, bind
        )

    def get_sink(self) -> SinkCallable:
        schema = ENCODING_REGISTRY['VideoFrame']
        eos_schema = ENCODING_REGISTRY['EndOfStream']
        context = zmq.Context()
        output_zmq_socket = context.socket(self.socket_type.value)
        if self.bind:
            output_zmq_socket.bind(self.socket)
        else:
            output_zmq_socket.connect(self.socket)

        def send_message(
            msg: SinkMessage,
            **kwargs,
        ):
            if isinstance(msg, SinkVideoFrame):
                message = {
                    'source_id': msg.frame_meta.source_id,
                    'pts': msg.frame_meta.pts,
                    'duration': msg.frame_meta.duration,
                    'framerate': msg.frame_meta.framerate,
                    'width': msg.frame_width,
                    'height': msg.frame_height,
                    'codec': msg.frame_codec.name if msg.frame_codec else None,
                    'frame': msg.frame,
                    'metadata': msg.frame_meta.metadata,
                    'tags': msg.frame_meta.tags,
                    'keyframe': msg.keyframe,
                }
                message_bin = serialize(schema, message)
                output_zmq_socket.send(message_bin)
            elif isinstance(msg, SinkEndOfStream):
                message = {'source_id': msg.source_id}
                message_bin = serialize(eos_schema, message)
                output_zmq_socket.send(message_bin)

        return send_message


@SINK_REGISTRY.register('console_sink')
class ConsoleSinkFactory(SinkFactory):
    """Just output messages to STDOUT."""

    def __init__(self, json_mode=False):
        self.json_mode = json_mode

    def get_sink(self) -> SinkCallable:
        if self.json_mode:

            def send_message(
                msg: SinkMessage,
                **kwargs,
            ):
                if not isinstance(msg, SinkVideoFrame):
                    return
                if 'objects' in msg.frame_meta.metadata:
                    for obj in msg.frame_meta.metadata['objects']:
                        for key, val in obj['bbox'].items():
                            obj['bbox'][key] = round(val, ndigits=3)
                        obj['confidence'] = round(obj['confidence'], ndigits=5)

                        for attr in obj['attributes']:
                            if isinstance(attr['value'], tuple):
                                attr['value'] = f"tuple len {len(attr['value'])}"
                            elif isinstance(attr['value'], list):
                                attr['value'] = f"list len {len(attr['value'])}"
                            elif isinstance(attr['value'], float):
                                attr['value'] = round(attr['value'], ndigits=3)
                            if isinstance(attr['confidence'], float):
                                attr['confidence'] = round(
                                    attr['confidence'], ndigits=5
                                )
                print(json.dumps(asdict(msg.frame_meta), indent=4))

        else:

            def send_message(
                msg: SinkMessage,
                **kwargs,
            ):
                if isinstance(msg, SinkVideoFrame):
                    message = f'Frame shape(WxH): {msg.frame_width}x{msg.frame_height}'
                    if msg.frame_codec is not None:
                        message += f', codec: {msg.frame_codec.name}'
                    if msg.frame is not None:
                        message += f', size (bytes): {len(msg.frame)}'
                    message += f'. Meta: {msg.frame_meta}.'
                    print(message)

                elif isinstance(msg, SinkEndOfStream):
                    message = f'End of stream: {msg.source_id}'
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

    def write(self, message: Dict):
        """Write message to a file in json format."""
        if not self._first:
            self._json_file.write(',')
        self._json_file.write(json.dumps(message))
        self._first = False

    def __del__(self):
        self._json_file.write(']')
        self._json_file.close()


@SINK_REGISTRY.register('file_sink')
class FileSinkFactory(SinkFactory):
    """Sink that outputs messages to json file.

    :param file_path: path to output file.
    """

    def __init__(self, file_path: str):
        self._json_writer = JsonWriter(file_path)

    def get_sink(self) -> SinkCallable:
        def send_message(
            msg: SinkMessage,
            **kwargs,
        ):
            if not isinstance(msg, SinkVideoFrame):
                return
            self._json_writer.write(asdict(msg.frame_meta))

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


def sink_factory(sink: Union[PipelineElement, List[PipelineElement]]) -> SinkCallable:
    """Init sink from config."""
    if isinstance(sink, PipelineElement):
        return SINK_REGISTRY.get(sink.element.lower())(**sink.properties).get_sink()

    sink_factories = []
    for _sink in sink:
        sink_factories.append(
            SINK_REGISTRY.get(_sink.element.lower())(**_sink.properties)
        )
    return MultiSinkFactory(*sink_factories).get_sink()
