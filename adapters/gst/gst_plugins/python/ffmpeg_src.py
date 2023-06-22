"""FFmpeg src."""
import inspect
from typing import Dict, NamedTuple, Optional

from ffmpeg_input import FFMpegSource, FFmpegLogLevel, VideoFrameEnvelope

from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec
from savant.gstreamer.utils import propagate_gst_setting_error, required_property
from savant.utils.logging import LoggerMixin

DEFAULT_QUEUE_LEN = 100
DEFAULT_FFMPEG_LOG_LEVEL = 'info'
STR_TO_FFMPEG_LOG_LEVEL = {
    name.lower(): getattr(FFmpegLogLevel, name)
    for name in dir(FFmpegLogLevel)
    if not name.startswith('_')
}

FFMPEG_SRC_PROPERTIES = {
    'uri': (
        str,
        'Source URI',
        'Source URI (e.g. "rtsp://localhost/stream", "/dev/video0").',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'params': (
        str,
        'FFmpeg params',
        'FFmpeg params. Comma separated string "key=value" '
        '(e.g. "rtsp_transport=tcp,c:v=v4l2m2m").',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'queue-len': (
        int,
        'Queue length',
        'Queue length.',
        0,
        GObject.G_MAXINT,
        DEFAULT_QUEUE_LEN,
        GObject.ParamFlags.READWRITE,
    ),
    'decode': (
        bool,
        'Decode frames',
        'Decode frames.',
        False,
        GObject.ParamFlags.READWRITE,
    ),
    'loglevel': (
        str,
        'FFmpeg log level',
        'FFmpeg log level.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
}


class FrameParams(NamedTuple):
    """Frame parameters."""

    codec_name: str
    width: str
    height: str
    framerate: str


class FFmpegSrc(LoggerMixin, GstBase.BaseSrc):
    """FFmpeg src GstPlugin."""

    GST_PLUGIN_NAME = 'ffmpeg_src'

    __gstmetadata__ = (
        'FFmpeg source',
        'Source',
        'Reads binary frames using FFmpeg',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = Gst.PadTemplate.new(
        'src',
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.new_any(),
    )

    __gproperties__ = FFMPEG_SRC_PROPERTIES

    def __init__(self):
        GstBase.BaseSrc.__init__(self)

        # properties
        self._uri: Optional[str] = None
        self._params: Dict[str, str] = {}
        self._queue_len: int = DEFAULT_QUEUE_LEN
        self._decode: bool = False
        self._loglevel: str = DEFAULT_FFMPEG_LOG_LEVEL

        self._frame_params: Optional[FrameParams] = None
        self._ffmpeg_source: Optional[FFMpegSource] = None

    def do_get_property(self, prop: GObject.GParamSpec):
        """Gst plugin get property function.

        :param prop: property parameters
        """

        if prop.name == 'uri':
            return self._uri
        if prop.name == 'params':
            if not self._params:
                return None
            return ','.join(f'{k}={v}' for k, v in self._params.items())
        if prop.name == 'queue-len':
            return self._queue_len
        if prop.name == 'decode':
            return self._decode
        if prop.name == 'loglevel':
            return self._loglevel
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop: GObject.GParamSpec, value):
        """Gst plugin set property function.

        :param prop: property parameters
        :param value: new value for param, type dependents on param
        """

        self.logger.debug('Setting property "%s" to "%s".', prop.name, value)
        if prop.name == 'uri':
            self._uri = value
        elif prop.name == 'params':
            self._params = {k: v for k, v in (p.split('=') for p in value.split(','))}
        elif prop.name == 'queue-len':
            self._queue_len = value
        elif prop.name == 'decode':
            self._decode = value
        elif prop.name == 'loglevel':
            self._loglevel = value
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def do_start(self):
        """Gst plugin start function."""

        try:
            required_property(self._uri, 'uri')
            self.logger.info('Creating FFMpegSource.')
            self._ffmpeg_source = FFMpegSource(
                self._uri,
                params=self._params,
                queue_len=self._queue_len,
                decode=self._decode,
                ffmpeg_log_level=STR_TO_FFMPEG_LOG_LEVEL[self._loglevel.lower()],
            )

        except Exception as exc:
            self.logger.exception('Failed to start element: %s', exc, exc_info=True)
            frame = inspect.currentframe()
            propagate_gst_setting_error(self, frame, __file__, text=exc.args[0])
            return False

        return True

    def do_create(self, offset: int, size: int, buffer: Gst.Buffer = None):
        """Create gst buffer."""

        self.logger.debug('Receiving next frame')

        frame: VideoFrameEnvelope = self._ffmpeg_source.video_frame()
        self.logger.debug(
            'Received frame with codec %s, PTS %s and DTS %s',
            frame.codec,
            frame.pts,
            frame.dts,
        )
        frame_params = FrameParams(
            codec_name=frame.codec,
            width=frame.frame_width,
            height=frame.frame_width,
            framerate=frame.fps,
        )
        if self._frame_params != frame_params:
            self.on_frame_params_change(frame_params)
        buffer: Gst.Buffer = Gst.Buffer.new_wrapped(frame.payload_as_bytes())
        tb_num, tb_denum = frame.time_base
        buffer.pts = frame.pts * tb_num * Gst.SECOND // tb_denum
        buffer.dts = frame.dts * tb_num * Gst.SECOND // tb_denum

        self.logger.debug(
            'Pushing buffer of size %s with PTS=%s, DTS=%s and duration=%s to src pad.',
            buffer.get_size(),
            buffer.pts,
            buffer.dts,
            buffer.duration,
        )

        return Gst.FlowReturn.OK, buffer

    def on_frame_params_change(self, frame_params: FrameParams):
        """Change caps when video parameter changed."""

        self._frame_params = frame_params
        codec = CODEC_BY_NAME[frame_params.codec_name]
        caps_str = ','.join(
            [
                codec.value.caps_with_params,
                f'width={frame_params.width}',
                f'height={frame_params.height}',
                f'framerate={frame_params.framerate}',
            ]
        )
        caps = Gst.Caps.from_string(caps_str)
        self.logger.info('Setting caps to %s', caps)
        self.set_caps(caps)


# register plugin
GObject.type_register(FFmpegSrc)
__gstelementfactory__ = (
    FFmpegSrc.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    FFmpegSrc,
)
