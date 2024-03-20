import inspect
from datetime import datetime
from typing import Any, Optional

import cv2

from adapters.ds.sinks.always_on_rtsp.last_frame import Frame, LastFrame, LastFrameRef
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.utils import (
    gst_post_library_settings_error,
    gst_post_stream_failed_error,
)
from savant.utils.logging import LoggerMixin

try:
    from savant.deepstream.opencv_utils import nvds_to_gpu_mat
except ImportError:
    nvds_to_gpu_mat = None

CAPS = Gst.Caps.from_string(
    'video/x-raw(memory:NVMM), format=RGBA; video/x-raw, format=RGBA'
)


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
            'Reference to last frame with its timestamp and resolution.',
            'Reference to last frame with its timestamp and resolution.',
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        # properties
        self._last_frame: Optional[LastFrameRef] = None

        self._use_gpu = nvds_to_gpu_mat is not None
        self._width = 0
        self._height = 0

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
            gst_post_library_settings_error(self, frame, __file__)
            return False
        return True

    def do_set_caps(self, caps: Gst.Caps):
        """Parse frame resolution from caps."""

        self.logger.debug('Sink caps changed to %s', caps)
        struct: Gst.Structure = caps.get_structure(0)
        self._width = struct.get_int('width').value
        self._height = struct.get_int('height').value
        self.logger.info('Frame resolution is %sx%s', self._width, self._height)

        return True

    def do_render(self, buffer: Gst.Buffer):
        try:
            if self._width == 0 or self._height == 0:
                raise RuntimeError('Frame resolution is not set')
            if self._use_gpu:
                with nvds_to_gpu_mat(buffer, batch_id=0) as frame:
                    last_frame = self._process_frame(frame)
            else:
                frame = buffer.extract_dup(0, buffer.get_size())
                last_frame = self._process_frame(frame)
            self._last_frame.frame = last_frame
        except Exception as e:
            error = f'Failed to process frame with PTS {buffer.pts}: {e}'
            self.logger.exception(error)
            frame = inspect.currentframe()
            gst_post_stream_failed_error(self, frame, __file__, error)
            return Gst.FlowReturn.ERROR

        return Gst.FlowReturn.OK

    def _process_frame(self, frame: Frame):
        self.logger.debug('Input frame resolution is %sx%s', self._width, self._height)
        if isinstance(frame, cv2.cuda.GpuMat):
            # Clone image for thread safety. The original CUDA memory will be released in this thread.
            # TODO: don't allocate CUDA memory if frame size wasn't changed (threadsafe?)
            content = frame.clone()
        else:
            content = frame

        return LastFrame(
            timestamp=datetime.now(),
            width=self._width,
            height=self._height,
            content=content,
        )


# register plugin
GObject.type_register(AlwaysOnRtspFrameSink)
__gstelementfactory__ = (
    AlwaysOnRtspFrameSink.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    AlwaysOnRtspFrameSink,
)
