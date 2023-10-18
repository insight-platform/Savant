import inspect
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import cv2

from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrame
from adapters.ds.sinks.always_on_rtsp.timestamp_overlay import TimestampOverlay
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.utils import gst_post_library_settings_error
from savant.utils.logging import LoggerMixin


class Mode(Enum):
    SCALE_TO_FIT = 'scale-to-fit'
    CROP_TO_FIT = 'crop-to-fit'


CAPS = Gst.Caps.from_string('video/x-raw(memory:NVMM), format=RGBA')
DEFAULT_MAX_DELAY = timedelta(seconds=1)
DEFAULT_MODE = Mode.SCALE_TO_FIT


class AlwaysOnRtspFrameProcessor(LoggerMixin, GstBase.BaseTransform):
    GST_PLUGIN_NAME: str = 'always_on_rtsp_frame_processor'

    __gstmetadata__ = (
        'Always-On-RTSP frame processor',
        'Transform',
        'Frame processor for Always-On-RTSP sink. '
        'Places stub image when actual frame is not available.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new('src', Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, CAPS),
        Gst.PadTemplate.new(
            'sink', Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, CAPS
        ),
    )

    __gproperties__ = {
        'max-delay-ms': (
            int,
            'Maximum delay for the last frame in milliseconds.',
            'Maximum delay for the last frame in milliseconds.',
            1,
            2147483647,
            int(DEFAULT_MAX_DELAY.total_seconds() * 1000),
            GObject.ParamFlags.READWRITE,
        ),
        'last-frame': (
            object,
            'Last frame with its timestamp.',
            'Last frame with its timestamp.',
            GObject.ParamFlags.READWRITE,
        ),
        'mode': (
            str,
            'Transfer mode.',
            'Transfer mode (allowed: ' f'{", ".join([mode.value for mode in Mode])}).',
            DEFAULT_MODE.value,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        # properties
        self._max_delay = DEFAULT_MAX_DELAY
        self._mode = DEFAULT_MODE
        self._last_frame: LastFrame = None

        self._time_overlay = TimestampOverlay()
        self._transfer = {
            Mode.SCALE_TO_FIT: self.scale_to_fit,
            Mode.CROP_TO_FIT: self.crop_to_fit,
        }

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        if prop.name == 'max-delay-ms':
            return int(self._max_delay.total_seconds() * 1000)
        if prop.name == 'last-frame':
            return self._last_frame
        if prop.name == 'mode':
            return self._mode.value
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        if prop.name == 'max-delay-ms':
            self._max_delay = timedelta(milliseconds=value)
        elif prop.name == 'last-frame':
            self._last_frame = value
        elif prop.name == 'mode':
            self._mode = Mode(value)
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        if self._last_frame is None:
            self.logger.exception('Property "last-frame" is not set')
            frame = inspect.currentframe()
            gst_post_library_settings_error(self, frame, __file__)
            return False
        return True

    def do_transform_ip(self, buffer: Gst.Buffer):
        with nvds_to_gpu_mat(buffer, batch_id=0) as output_frame:
            self.logger.debug('Output frame resolution is %sx%s', *output_frame.size())
            now = datetime.now()
            input_frame = self._last_frame.frame
            timestamp = self._last_frame.timestamp
            delay = now - timestamp
            if input_frame is not None and delay < self._max_delay:
                self.logger.debug(
                    'Got frame with timestamp %s and resolution %sx%s. Frame delay is %s.',
                    timestamp,
                    *input_frame.size(),
                    delay,
                )
                if input_frame.size() == output_frame.size():
                    input_frame.copyTo(output_frame)
                else:
                    self._transfer[self._mode](input_frame, output_frame)
            else:
                self.logger.debug(
                    'No new data received from the input. Sending stub image with the timestamp.'
                )
                self._time_overlay.overlay_timestamp(output_frame, now)
        return Gst.FlowReturn.OK

    def scale_to_fit(
        self,
        input_frame: cv2.cuda.GpuMat,
        output_frame: cv2.cuda.GpuMat,
    ):
        in_width, in_height = input_frame.size()
        in_aspect_ratio = in_width / in_height
        out_width, out_height = output_frame.size()
        out_aspect_ratio = out_width / out_height
        if in_aspect_ratio < out_aspect_ratio:
            target_height = out_height
            target_width = int(target_height * in_aspect_ratio)
        else:
            target_width = out_width
            target_height = int(target_width / in_aspect_ratio)
        self.logger.debug(
            'Scaling input image from %sx%s to %sx%s',
            in_width,
            in_height,
            target_width,
            target_height,
        )
        output_frame.setTo((0, 0, 0, 0))
        target = cv2.cuda.GpuMat(
            output_frame,
            (
                (out_width - target_width) // 2,  # left
                (out_height - target_height) // 2,  # top
                target_width,  # width
                target_height,  # height
            ),
        )
        cv2.cuda.resize(input_frame, (target_width, target_height), target)

    def crop_to_fit(
        self,
        input_frame: cv2.cuda.GpuMat,
        output_frame: cv2.cuda.GpuMat,
    ):
        in_width, in_height = input_frame.size()
        out_width, out_height = output_frame.size()
        target_width = min(in_width, out_width)
        target_height = min(in_height, out_height)

        self.logger.debug(
            'Cropping input image from %sx%s to %sx%s',
            in_width,
            in_height,
            target_width,
            target_height,
        )
        output_frame.setTo((0, 0, 0, 0))
        source = cv2.cuda.GpuMat(
            input_frame,
            (
                (in_width - target_width) // 2,  # left
                (in_height - target_height) // 2,  # top
                target_width,  # width
                target_height,  # height
            ),
        )
        target = cv2.cuda.GpuMat(
            output_frame,
            (
                (out_width - target_width) // 2,  # left
                (out_height - target_height) // 2,  # top
                target_width,  # width
                target_height,  # height
            ),
        )
        source.copyTo(target)


# register plugin
GObject.type_register(AlwaysOnRtspFrameProcessor)
__gstelementfactory__ = (
    AlwaysOnRtspFrameProcessor.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    AlwaysOnRtspFrameProcessor,
)
