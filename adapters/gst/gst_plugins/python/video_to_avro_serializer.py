import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Union

from savant.api import serialize, ENCODING_REGISTRY
from savant.api.enums import ExternalFrameType
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.codecs import CODEC_BY_CAPS_NAME, Codec
from savant.gstreamer.metadata import DEFAULT_FRAMERATE
from savant.gstreamer.utils import LoggerMixin

EMBEDDED_FRAME_TYPE = 'embedded'


class FrameParams(NamedTuple):
    """Frame parameters."""

    codec_name: str
    width: str
    height: str
    framerate: str


class VideoToAvroSerializer(LoggerMixin, GstBase.BaseTransform):
    """GStreamer plugin to serialize video frames to avro message."""

    GST_PLUGIN_NAME: str = 'video_to_avro_serializer'

    __gstmetadata__ = (
        'Serializes video frames to avro messages',
        'Transform',
        'Serializes video frame to avro message',
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
        self.schema = ENCODING_REGISTRY["VideoFrame"]
        self.eos_schema = ENCODING_REGISTRY["EndOfStream"]
        # properties
        self.source_id: Optional[str] = None
        self.eos_on_file_end: bool = True
        self.eos_on_frame_params_change: bool = True
        # will be set after caps negotiation
        self.frame_params: Optional[FrameParams] = None
        self.last_frame_params: Optional[FrameParams] = None
        self.location: Optional[Path] = None
        self.last_location: Optional[Path] = None
        self.default_framerate: str = DEFAULT_FRAMERATE
        self.frame_type: Optional[ExternalFrameType] = ExternalFrameType.ZEROMQ

        self.stream_in_progress = False
        self.read_metadata: bool = False
        self.json_metadata = None

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
            ):
                self.json_metadata = self.read_json_metadata_file(
                    self.location.parent / f"{self.location.stem}.json"
                )
                self.send_end_message()
        self.last_location = self.location
        self.last_frame_params = self.frame_params

        frame_mapinfo: Optional[Gst.MapInfo] = None
        if self.frame_type is None:
            result, frame_mapinfo = in_buf.map(Gst.MapFlags.READ)
            assert result, 'Cannot read buffer.'
            frame = frame_mapinfo.data
        elif self.frame_type == ExternalFrameType.ZEROMQ:
            frame = {'type': self.frame_type.value}
        else:
            self.logger.error('Unsupported frame type "%s".', self.frame_type.value)
            return Gst.FlowReturn.ERROR
        message = self.build_message(
            in_buf.pts,
            in_buf.dts if in_buf.dts != Gst.CLOCK_TIME_NONE else None,
            in_buf.duration if in_buf.duration != Gst.CLOCK_TIME_NONE else None,
            frame=frame,
            keyframe=not in_buf.has_flags(Gst.BufferFlags.DELTA_UNIT),
        )
        frame_meta = serialize(self.schema, message)

        out_buf: Gst.Buffer = Gst.Buffer.new_wrapped(frame_meta)
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
                self.json_metadata = self.read_json_metadata_file(
                    self.location.parent / f"{self.location.stem}.json"
                )

        # Cannot use `super()` since it is `self`
        return GstBase.BaseTransform.do_sink_event(self, event)

    def read_json_metadata_file(self, location: Path):
        json_metadata = None
        if self.read_metadata:
            if location.is_file():
                with open(location, 'r') as fp:
                    json_metadata = dict(
                        map(
                            lambda x: (x["pts"], x),
                            filter(
                                lambda x: x["schema"] == "VideoFrame",
                                map(json.loads, fp.readlines()),
                            ),
                        )
                    )
            else:
                self.logger.warning('JSON file `%s` not found', location.absolute())
        return json_metadata

    def send_end_message(self):
        data = serialize(
            self.eos_schema,
            {'source_id': self.source_id},
        )
        out_buf = Gst.Buffer.new_wrapped(data)
        self.srcpad.push(out_buf)
        self.stream_in_progress = False

    def build_message(
        self,
        pts: int,
        dts: Optional[int],
        duration: Optional[int],
        frame: Optional[Union[bytes, Dict[str, Any]]],
        **kwargs,
    ):
        if pts == Gst.CLOCK_TIME_NONE:
            # TODO: support CLOCK_TIME_NONE in schema
            pts = 0
        frame_metadata = None
        if self.read_metadata and self.json_metadata:
            frame_metadata = self.json_metadata[pts]
        message = {
            'source_id': self.source_id,
            'framerate': self.frame_params.framerate,
            'width': self.frame_params.width,
            'height': self.frame_params.height,
            'codec': self.frame_params.codec_name,
            'pts': pts,
            'dts': dts,
            'duration': duration,
            'frame': frame,
            'metadata': frame_metadata["metadata"] if self.read_metadata else None,
            **kwargs,
        }
        if self.location:
            message['tags'] = {'location': str(self.location)}
        return message


# register plugin
GObject.type_register(VideoToAvroSerializer)
__gstelementfactory__ = (
    VideoToAvroSerializer.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    VideoToAvroSerializer,
)
