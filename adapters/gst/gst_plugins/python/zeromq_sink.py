"""ZeroMQ sink."""

import inspect
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from savant_rs.primitives import (
    AttributeValue,
    EndOfStream,
    Shutdown,
    VideoFrame,
    VideoFrameContent,
    VideoFrameTransformation,
)
from savant_rs.utils.serialization import Message
from savant_rs.zmq import (
    BlockingWriter,
    WriterConfigBuilder,
    WriterResultAck,
    WriterResultSuccess,
)
from splitstream import splitfile

from gst_plugins.python.zeromq_properties import ZEROMQ_PROPERTIES, socket_type_property
from savant.api.builder import add_objects_to_video_frame
from savant.api.constants import DEFAULT_FRAMERATE, DEFAULT_NAMESPACE, DEFAULT_TIME_BASE
from savant.api.enums import ExternalFrameType
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.codecs import CODEC_BY_CAPS_NAME, Codec
from savant.gstreamer.event import parse_savant_frame_tags_event
from savant.gstreamer.utils import (
    gst_post_library_settings_error,
    gst_post_stream_failed_error,
    required_property,
)
from savant.utils.logging import LoggerMixin
from savant.utils.zeromq import Defaults, SenderSocketTypes, get_zmq_socket_uri_options

EMBEDDED_FRAME_TYPE = 'embedded'
DEFAULT_SOURCE_ID_PATTERN = 'source-%d'


class FrameParams(NamedTuple):
    """Frame parameters."""

    codec_name: str
    width: int
    height: int
    framerate: str


def is_videoframe_metadata(metadata: Dict[str, Any]) -> bool:
    """Check that metadata contained if metadata is a video frame metadata. ."""
    if 'schema' in metadata and metadata['schema'] != 'VideoFrame':
        return False
    return True


class ZeroMQSink(LoggerMixin, GstBase.BaseSink):
    """Serializes video stream to savant-rs message and sends it to ZeroMQ socket."""

    GST_PLUGIN_NAME = 'zeromq_sink'

    __gstmetadata__ = (
        'Serializes video stream to savant-rs message and sends it to ZeroMQ socket.',
        'Sink',
        'Serializes video stream to savant-rs message and sends it to ZeroMQ socket.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = Gst.PadTemplate.new(
        'sink',
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string(';'.join(x.value.caps_with_params for x in Codec)),
    )

    __gproperties__ = {
        **ZEROMQ_PROPERTIES,
        'socket-type': socket_type_property(SenderSocketTypes),
        'send-hwm': (
            int,
            'High watermark for outbound messages',
            'High watermark for outbound messages',
            1,
            GObject.G_MAXINT,
            Defaults.SEND_HWM,
            GObject.ParamFlags.READWRITE,
        ),
        'receive-timeout': (
            int,
            'Receive timeout socket option',
            'Receive timeout socket option',
            0,
            GObject.G_MAXINT,
            Defaults.SENDER_RECEIVE_TIMEOUT,
            GObject.ParamFlags.READWRITE,
        ),
        'receive-retries': (
            int,
            'Retries to receive confirmation message',
            'Retries to receive confirmation message',
            1,
            GObject.G_MAXINT,
            Defaults.RECEIVE_RETRIES,
            GObject.ParamFlags.READWRITE,
        ),
        'source-id': (
            str,
            'Source ID',
            'Source ID, e.g. "camera1".',
            None,
            GObject.ParamFlags.READWRITE | Gst.PARAM_MUTABLE_READY,
        ),
        'location': (
            str,
            'Source location',
            'Source location',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        # TODO: make fraction
        'framerate': (
            str,
            'Default framerate',
            'Default framerate',
            DEFAULT_FRAMERATE,
            GObject.ParamFlags.READWRITE,
        ),
        'eos-on-file-end': (
            bool,
            'Send EOS at the end of each file',
            'Send EOS at the end of each file',
            True,
            GObject.ParamFlags.READWRITE,
        ),
        'eos-on-loop-end': (
            bool,
            'Send EOS on a loop end',
            'Send EOS on a loop end',
            False,
            GObject.ParamFlags.READWRITE,
        ),
        'eos-on-frame-params-change': (
            bool,
            'Send EOS when frame parameters changed',
            'Send EOS when frame parameters changed',
            True,
            GObject.ParamFlags.READWRITE,
        ),
        'read-metadata': (
            bool,
            'Read metadata',
            'Attempt to read the metadata of objects from the JSON file that has the identical name '
            'as the source file with `json` extension, and then send it to the module.',
            False,
            GObject.ParamFlags.READWRITE,
        ),
        'frame-type': (
            str,
            'Frame type.',
            'Frame type (allowed: '
            f'{", ".join([EMBEDDED_FRAME_TYPE] + [enum_member.value for enum_member in ExternalFrameType])})',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'enable-multistream': (
            bool,
            'Enable multistream',
            'Enable multistream',
            False,
            GObject.ParamFlags.READWRITE | Gst.PARAM_MUTABLE_READY,
        ),
        'source-id-pattern': (
            str,
            'Pattern for source ID',
            'Pattern for source ID when multistream is enabled. E.g. "source-%d".',
            DEFAULT_SOURCE_ID_PATTERN,
            GObject.ParamFlags.READWRITE | Gst.PARAM_MUTABLE_READY,
        ),
        'number-of-streams': (
            int,
            'Number of streams',
            'Number of streams',
            1,  # min
            1024,  # max
            1,
            GObject.ParamFlags.READWRITE | Gst.PARAM_MUTABLE_READY,
        ),
        'shutdown-auth': (
            str,
            'Authentication key for Shutdown message.',
            'Authentication key for Shutdown message. When specified, a shutdown'
            'message will be sent at the end of the stream.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        GstBase.BaseSink.__init__(self)

        # properties
        self.socket: str = None
        self.socket_type: str = SenderSocketTypes.DEALER.name
        self.bind: bool = True
        self.source_id: Optional[str] = None
        self.eos_on_file_end: bool = True
        self.eos_on_loop_end: bool = False
        self.eos_on_frame_params_change: bool = True
        self.enable_multistream: bool = False
        self.source_id_pattern: str = DEFAULT_SOURCE_ID_PATTERN
        self.number_of_streams: int = 1
        self.shutdown_auth: Optional[str] = None
        self.send_hwm = Defaults.SEND_HWM

        # will be set after caps negotiation
        self.frame_params: Optional[FrameParams] = None
        self.initial_size_transformation: Optional[VideoFrameTransformation] = None
        self.last_frame_params: Optional[FrameParams] = None
        self.location: Optional[Path] = None
        self.last_location: Optional[Path] = None
        self.new_loop: bool = False
        self.default_framerate: str = DEFAULT_FRAMERATE

        self.source_ids: List[str] = []
        self.stream_in_progress = False
        self.read_metadata: bool = False
        self.json_metadata = None
        self.frame_num = 0
        self.writer: BlockingWriter = None
        self.receive_timeout = Defaults.SENDER_RECEIVE_TIMEOUT
        self.receive_retries = Defaults.RECEIVE_RETRIES
        self.savant_frame_tags: Dict[str, str] = {}
        self.set_sync(False)

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates
            the metadata required to specify parameters
        """

        if prop.name == 'socket':
            return self.socket
        if prop.name == 'socket-type':
            return self.socket_type
        if prop.name == 'bind':
            return self.bind
        if prop.name == 'send-hwm':
            return self.send_hwm
        if prop.name == 'receive-timeout':
            return self.receive_timeout
        if prop.name == 'receive-retries':
            return self.receive_retries

        if prop.name == 'source-id':
            return self.source_id
        if prop.name == 'location':
            return self.location
        if prop.name == 'framerate':
            return self.default_framerate
        if prop.name == 'eos-on-file-end':
            return self.eos_on_file_end
        if prop.name == 'eos-on-loop-end':
            return self.eos_on_loop_end
        if prop.name == 'eos-on-frame-params-change':
            return self.eos_on_frame_params_change
        if prop.name == 'read-metadata':
            return self.read_metadata
        if prop.name == 'enable-multistream':
            return self.enable_multistream
        if prop.name == 'source-id-pattern':
            return self.source_id_pattern
        if prop.name == 'number-of-streams':
            return self.number_of_streams
        if prop.name == 'shutdown-auth':
            return self.shutdown_auth

        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: structure that encapsulates
            the metadata required to specify parameters
        :param value: new value for param, type dependents on param
        """

        self.logger.debug('Setting property "%s" to "%s".', prop.name, value)
        if prop.name == 'socket':
            self.socket = value
        elif prop.name == 'socket-type':
            self.socket_type = value
        elif prop.name == 'bind':
            self.bind = value
        elif prop.name == 'send-hwm':
            self.send_hwm = value
        elif prop.name == 'receive-timeout':
            self.receive_timeout = value
        elif prop.name == 'receive-retries':
            self.receive_retries = value

        elif prop.name == 'source-id':
            self.source_id = value
        elif prop.name == 'location':
            self.location = value
        elif prop.name == 'framerate':
            try:
                Fraction(value)  # validate
            except (ZeroDivisionError, ValueError) as e:
                raise AttributeError(f'Invalid property {prop.name}: {e}.') from e
            self.default_framerate = value
        elif prop.name == 'eos-on-file-end':
            self.eos_on_file_end = value
        elif prop.name == 'eos-on-loop-end':
            self.eos_on_loop_end = value
        elif prop.name == 'eos-on-frame-params-change':
            self.eos_on_frame_params_change = value
        elif prop.name == 'read-metadata':
            self.read_metadata = value
        elif prop.name == 'enable-multistream':
            self.enable_multistream = value
        elif prop.name == 'source-id-pattern':
            self.source_id_pattern = value
        elif prop.name == 'number-of-streams':
            self.number_of_streams = value
        elif prop.name == 'shutdown-auth':
            self.shutdown_auth = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def get_source_ids(self) -> Optional[List[str]]:
        if self.enable_multistream:
            if self.source_id_pattern is None:
                self.logger.error(
                    'Source ID pattern is required when enable-multistream=true.'
                )
                return None
            try:
                source_ids = [
                    self.source_id_pattern % i for i in range(self.number_of_streams)
                ]
            except TypeError as e:
                self.logger.error('Invalid source ID pattern: %s', e)
                return None
            if len(source_ids) != len(set(source_ids)):
                self.logger.error(
                    'Duplicate source IDs. Check source-id-pattern property.'
                )
                return None

            return source_ids

        if self.source_id is None:
            self.logger.error('Source ID is required when enable-multistream=false.')
            return None
        return [self.source_id]

    def do_start(self):
        """Start sink."""

        self.source_ids = self.get_source_ids()
        if self.source_ids is None:
            return False

        try:
            required_property('socket', self.socket)
            config_builder = WriterConfigBuilder(self.socket)
            if not get_zmq_socket_uri_options(self.socket):
                config_builder.with_socket_type(
                    SenderSocketTypes[self.socket_type].value
                )
                config_builder.with_bind(self.bind)
            config_builder.with_send_hwm(self.send_hwm)
            config_builder.with_receive_timeout(self.receive_timeout)
            config_builder.with_receive_retries(self.receive_retries)
            config_builder.with_send_timeout(self.receive_timeout)
            config_builder.with_send_retries(self.receive_retries)
            self.writer = BlockingWriter(config_builder.build())
            self.writer.start()

        except Exception as exc:
            error = f'Failed to start ZeroMQ sink with socket {self.socket}: {exc}.'
            self.logger.exception(error, exc_info=True)
            frame = inspect.currentframe()
            gst_post_library_settings_error(self, frame, __file__, error)
            # prevents pipeline from starting
            return False

        return True

    def do_set_caps(self, caps: Gst.Caps):
        """Checks caps after negotiations."""

        self.logger.info('Sink caps changed to %s', caps)
        struct: Gst.Structure = caps.get_structure(0)
        try:
            codec = CODEC_BY_CAPS_NAME[struct.get_name()]
        except KeyError:
            self.logger.error('Not supported caps: %s', caps.to_string())
            return False
        codec_name = codec.value.name
        frame_width = struct.get_int('width').value
        frame_height = struct.get_int('height').value
        if struct.has_field('framerate'):
            _, framerate_num, framerate_demon = struct.get_fraction('framerate')
            framerate = f'{framerate_num}/{framerate_demon}'
        else:
            framerate = self.default_framerate
        self.frame_params = FrameParams(
            codec_name=codec_name,
            width=frame_width,
            height=frame_height,
            framerate=framerate,
        )
        self.initial_size_transformation = VideoFrameTransformation.initial_size(
            frame_width,
            frame_height,
        )

        return True

    def do_render(self, buffer: Gst.Buffer):
        """Send data through ZeroMQ."""
        self.logger.debug(
            'Processing frame %s of size %s', buffer.pts, buffer.get_size()
        )
        if self.stream_in_progress:
            if (
                self.eos_on_file_end
                and self.location != self.last_location
                or self.eos_on_frame_params_change
                and self.frame_params != self.last_frame_params
                or self.eos_on_loop_end
                and self.new_loop
            ):
                self.json_metadata = self.read_json_metadata_file(
                    self.location.parent / f'{self.location.stem}.json'
                )
                self.frame_num = 0
                self.send_eos()
        self.last_location = self.location
        self.last_frame_params = self.frame_params
        self.new_loop = False
        self.stream_in_progress = True

        content = buffer.extract_dup(0, buffer.get_size())
        base_frame = self.build_video_frame(
            source_id=self.source_ids[0][0],
            pts=buffer.pts,
            dts=buffer.dts if buffer.dts != Gst.CLOCK_TIME_NONE else None,
            duration=(
                buffer.duration if buffer.duration != Gst.CLOCK_TIME_NONE else None
            ),
            keyframe=not buffer.has_flags(Gst.BufferFlags.DELTA_UNIT),
        )

        for source_id in self.source_ids:
            frame = base_frame.copy()
            frame.source_id = source_id
            if not self.send_message_to_zmq(source_id, frame.to_message(), content):
                return Gst.FlowReturn.ERROR

        return Gst.FlowReturn.OK

    def do_stop(self):
        """Stop source."""

        if self.shutdown_auth is not None:
            self.logger.info('Sending serialized Shutdown message')
            self.send_message_to_zmq(
                self.source_ids[0],
                Shutdown(self.shutdown_auth).to_message(),
            )

        self.logger.info('Terminating ZeroMQ writer.')
        self.writer.shutdown()
        self.logger.info('ZeroMQ writer terminated')
        return True

    def do_event(self, event: Gst.Event):
        if event.type == Gst.EventType.EOS:
            self.logger.info('Got End-Of-Stream event')
            self.send_eos()

        elif event.type == Gst.EventType.TAG:
            tag_list: Gst.TagList = event.parse_tag()
            has_location, location = tag_list.get_string(Gst.TAG_LOCATION)
            if has_location:
                self.logger.info('Set location to %s', location)
                self.location = Path(location)
                self.new_loop = True
                self.json_metadata = self.read_json_metadata_file(
                    self.location.parent / f'{self.location.stem}.json'
                )
                self.frame_num = 0

        elif event.type == Gst.EventType.CUSTOM_DOWNSTREAM:
            tags = parse_savant_frame_tags_event(event)
            if tags is not None:
                self.savant_frame_tags = tags

        # Cannot use `super()` since it is `self`
        return GstBase.BaseSink.do_event(self, event)

    def build_video_frame(
        self,
        source_id: str,
        pts: int,
        dts: Optional[int],
        duration: Optional[int],
        keyframe: bool,
    ) -> VideoFrame:
        if pts == Gst.CLOCK_TIME_NONE:
            pts = 0
        objects = None
        if self.read_metadata and self.json_metadata:
            frame_metadata = self.json_metadata[self.frame_num]
            self.frame_num += 1
            objects = frame_metadata['objects']

        video_frame = VideoFrame(
            source_id=source_id,
            framerate=self.frame_params.framerate,
            width=self.frame_params.width,
            height=self.frame_params.height,
            codec=self.frame_params.codec_name,
            content=VideoFrameContent.external(ExternalFrameType.ZEROMQ.value, None),
            keyframe=keyframe,
            pts=pts,
            dts=dts,
            duration=duration,
            time_base=DEFAULT_TIME_BASE,
        )
        video_frame.add_transformation(self.initial_size_transformation)
        if objects:
            add_objects_to_video_frame(video_frame, objects)
        if self.location:
            video_frame.set_persistent_attribute(
                namespace=DEFAULT_NAMESPACE,
                name='location',
                values=[AttributeValue.string(str(self.location))],
            )
        for tag_name, tag_value in self.savant_frame_tags.items():
            if tag_name == 'location':
                continue
            video_frame.set_persistent_attribute(
                namespace=DEFAULT_NAMESPACE,
                name=tag_name,
                values=[AttributeValue.string(tag_value)],
            )

        return video_frame

    def read_json_metadata_file(self, location: Path):
        json_metadata = None
        if self.read_metadata:
            if location.is_file():
                with open(location, 'r') as fp:
                    json_metadata = list(
                        map(
                            lambda x: x['metadata'],
                            filter(
                                is_videoframe_metadata,
                                map(json.loads, splitfile(fp, format='json')),
                            ),
                        )
                    )
            else:
                self.logger.warning('JSON file `%s` not found', location.absolute())
        return json_metadata

    def send_eos(self):
        self.logger.info('Sending serialized EOS message')
        for source_id in self.source_ids:
            self.send_message_to_zmq(
                source_id,
                EndOfStream(source_id).to_message(),
            )
        self.stream_in_progress = False

    def send_message_to_zmq(
        self,
        source_id: str,
        message: Message,
        content: bytes = b'',
    ) -> bool:
        try:
            send_result = self.writer.send_message(source_id, message, content)
            if isinstance(send_result, (WriterResultAck, WriterResultSuccess)):
                return True
            error = f'Failed to send message to ZeroMQ: {send_result}'
        except Exception as exc:
            error = f'Failed to send message to ZeroMQ: {exc}'

        self.logger.error(error)
        frame = inspect.currentframe()
        gst_post_stream_failed_error(
            gst_element=self,
            frame=frame,
            file_path=__file__,
            text=error,
        )
        return False


# register plugin
GObject.type_register(ZeroMQSink)
__gstelementfactory__ = (
    ZeroMQSink.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    ZeroMQSink,
)
