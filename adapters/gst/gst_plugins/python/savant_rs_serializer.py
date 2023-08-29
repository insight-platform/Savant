import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    EndOfStream,
    VideoFrame,
    VideoFrameContent,
    VideoFrameTransformation,
)
from savant_rs.utils.serialization import Message, save_message_to_bytes
from splitstream import splitfile

from savant.api.builder import add_objects_to_video_frame
from savant.api.constants import DEFAULT_FRAMERATE, DEFAULT_NAMESPACE, DEFAULT_TIME_BASE
from savant.api.enums import ExternalFrameType
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.codecs import CODEC_BY_CAPS_NAME, Codec
from savant.gstreamer.utils import gst_buffer_from_list
from savant.utils.logging import LoggerMixin

EMBEDDED_FRAME_TYPE = 'embedded'


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


class SavantRsSerializer(LoggerMixin, GstBase.BaseTransform):
    """GStreamer plugin to serialize video stream to savant-rs message."""

    GST_PLUGIN_NAME: str = 'savant_rs_serializer'

    __gstmetadata__ = (
        'Serializes video stream to savant-rs messages',
        'Transform',
        'Serializes video stream to savant-rs message',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'src',
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        ),
        Gst.PadTemplate.new(
            'sink',
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(';'.join(x.value.caps_with_params for x in Codec)),
        ),
    )

    __gproperties__ = {
        'source-id': (
            str,
            'Source ID',
            'Source ID, e.g. "camera1".',
            None,
            GObject.ParamFlags.READWRITE,
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
    }

    def __init__(self):
        super().__init__()
        # properties
        self.source_id: Optional[str] = None
        self.zmq_topic: Optional[bytes] = None
        self.eos_on_file_end: bool = True
        self.eos_on_loop_end: bool = False
        self.eos_on_frame_params_change: bool = True
        # will be set after caps negotiation
        self.frame_params: Optional[FrameParams] = None
        self.initial_size_transformation: Optional[VideoFrameTransformation] = None
        self.last_frame_params: Optional[FrameParams] = None
        self.location: Optional[Path] = None
        self.last_location: Optional[Path] = None
        self.new_loop: bool = False
        self.default_framerate: str = DEFAULT_FRAMERATE
        self.frame_type: Optional[ExternalFrameType] = ExternalFrameType.ZEROMQ

        self.stream_in_progress = False
        self.read_metadata: bool = False
        self.json_metadata = None
        self.frame_num = 0

    def do_set_caps(  # pylint: disable=unused-argument
        self, in_caps: Gst.Caps, out_caps: Gst.Caps
    ):
        """Checks caps after negotiations."""
        self.logger.info('Sink caps changed to %s', in_caps)
        struct: Gst.Structure = in_caps.get_structure(0)
        try:
            codec = CODEC_BY_CAPS_NAME[struct.get_name()]
        except KeyError:
            self.logger.error('Not supported caps: %s', in_caps.to_string())
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

    def do_get_property(self, prop: GObject.GParamSpec):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
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
        if prop.name == 'frame-type':
            if self.frame_type is None:
                return EMBEDDED_FRAME_TYPE
            return self.frame_type.value
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        if prop.name == 'source-id':
            self.source_id = value
            self.zmq_topic = f'{value}/'.encode()
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
        elif prop.name == 'frame-type':
            if value == EMBEDDED_FRAME_TYPE:
                self.frame_type = None
            else:
                self.frame_type = ExternalFrameType(value)
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        assert self.source_id, 'Source ID is required.'
        return True

    def do_prepare_output_buffer(self, in_buf: Gst.Buffer):
        """Transform gst function."""

        self.logger.debug(
            'Processing frame %s of size %s', in_buf.pts, in_buf.get_size()
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
                self.send_end_message()
        self.last_location = self.location
        self.last_frame_params = self.frame_params
        self.new_loop = False

        frame_mapinfo: Optional[Gst.MapInfo] = None
        if self.frame_type is None:
            result, frame_mapinfo = in_buf.map(Gst.MapFlags.READ)
            assert result, 'Cannot read buffer.'
            content = VideoFrameContent.internal(frame_mapinfo.data)
        elif self.frame_type == ExternalFrameType.ZEROMQ:
            content = VideoFrameContent.external(self.frame_type.value, None)
        else:
            self.logger.error('Unsupported frame type "%s".', self.frame_type.value)
            return Gst.FlowReturn.ERROR

        frame = self.build_video_frame(
            in_buf.pts,
            in_buf.dts if in_buf.dts != Gst.CLOCK_TIME_NONE else None,
            in_buf.duration if in_buf.duration != Gst.CLOCK_TIME_NONE else None,
            content=content,
            keyframe=not in_buf.has_flags(Gst.BufferFlags.DELTA_UNIT),
        )
        message = Message.video_frame(frame)
        data = save_message_to_bytes(message)

        out_buf: Gst.Buffer = gst_buffer_from_list([self.zmq_topic, data])
        if frame_mapinfo is not None:
            in_buf.unmap(frame_mapinfo)
        else:
            out_buf.append_memory(in_buf.get_memory_range(0, -1))
        out_buf.pts = in_buf.pts
        out_buf.dts = in_buf.dts
        out_buf.duration = in_buf.duration
        self.stream_in_progress = True

        return Gst.FlowReturn.OK, out_buf

    def do_sink_event(self, event: Gst.Event):
        if event.type == Gst.EventType.EOS:
            self.logger.info('Got End-Of-Stream event')
            self.send_end_message()

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

        # Cannot use `super()` since it is `self`
        return GstBase.BaseTransform.do_sink_event(self, event)

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

    def send_end_message(self):
        self.logger.info('Sending serialized EOS message')
        message = Message.end_of_stream(EndOfStream(self.source_id))
        data = save_message_to_bytes(message)
        out_buf = gst_buffer_from_list([self.zmq_topic, data])
        self.srcpad.push(out_buf)
        self.stream_in_progress = False

    def build_video_frame(
        self,
        pts: int,
        dts: Optional[int],
        duration: Optional[int],
        content: VideoFrameContent,
        keyframe: bool,
    ) -> VideoFrame:
        if pts == Gst.CLOCK_TIME_NONE:
            # TODO: support CLOCK_TIME_NONE in schema
            pts = 0
        objects = None
        if self.read_metadata and self.json_metadata:
            frame_metadata = self.json_metadata[self.frame_num]
            self.frame_num += 1
            objects = frame_metadata['objects']

        video_frame = VideoFrame(
            source_id=self.source_id,
            framerate=self.frame_params.framerate,
            width=self.frame_params.width,
            height=self.frame_params.height,
            codec=self.frame_params.codec_name,
            content=content,
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
            video_frame.set_attribute(
                Attribute(
                    namespace=DEFAULT_NAMESPACE,
                    name='location',
                    values=[AttributeValue.string(str(self.location))],
                )
            )

        return video_frame


# register plugin
GObject.type_register(SavantRsSerializer)
__gstelementfactory__ = (
    SavantRsSerializer.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SavantRsSerializer,
)
