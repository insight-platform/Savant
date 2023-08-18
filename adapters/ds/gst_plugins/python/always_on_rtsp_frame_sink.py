import inspect
from datetime import datetime
from typing import Any

from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrame
from adapters.ds.sinks.always_on_rtsp.timestamp_overlay import TimestampOverlay
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.utils import propagate_gst_setting_error
from savant.utils.logging import LoggerMixin

CAPS = Gst.Caps.from_string('video/x-raw(memory:NVMM), format=RGBA')


class AlwaysOnRtspFrameSink(LoggerMixin, GstBase.BaseSink):
    GST_PLUGIN_NAME: str = 'always_on_rtsp_frame_sink'

    __gstmetadata__ = (
        'Always-On-RTSP frame sink',
        'Sink',
        'Frame sink for Always-On-RTSP sink. Takes decoded frames '
        'from the input and sends them to the output pipeline.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'sink', Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, CAPS
        ),
    )

    __gproperties__ = {
        'last-frame': (
            object,
            'Last frame with its timestamp.',
            'Last frame with its timestamp.',
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        # properties
        self._last_frame: LastFrame = None

        self._time_overlay = TimestampOverlay()

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        if prop.name == 'last-frame':
            return self._last_frame
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        if prop.name == 'last-frame':
            self._last_frame = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        if self._last_frame is None:
            self.logger.exception('Property "last-frame" is not set')
            frame = inspect.currentframe()
            propagate_gst_setting_error(self, frame, __file__)
            return False
        return True

    def do_render(self, buffer: Gst.Buffer):
        with nvds_to_gpu_mat(buffer, batch_id=0) as frame:
            self.logger.debug('Input frame resolution is %sx%s', *frame.size())
            # Clone image for thread safety. The original CUDA memory will be released in this thread.
            # TODO: don't allocate CUDA memory if frame size wasn't changed (threadsafe?)
            self._last_frame.frame = frame.clone()
            self._last_frame.timestamp = datetime.now()
        return Gst.FlowReturn.OK


# register plugin
GObject.type_register(AlwaysOnRtspFrameSink)
__gstelementfactory__ = (
    AlwaysOnRtspFrameSink.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    AlwaysOnRtspFrameSink,
)
