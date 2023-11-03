import inspect
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrame, LastFrameRef
from adapters.ds.sinks.always_on_rtsp.timestamp_overlay import TimestampOverlay
from adapters.ds.sinks.always_on_rtsp.utils import Frame
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.utils import (
    gst_post_library_settings_error,
    gst_post_stream_failed_error,
    map_gst_buffer,
)
from savant.utils.logging import LoggerMixin

try:
    from savant.deepstream.opencv_utils import nvds_to_gpu_mat
except ImportError:
    nvds_to_gpu_mat = None


class Mode(Enum):
    SCALE_TO_FIT = 'scale-to-fit'
    CROP_TO_FIT = 'crop-to-fit'


CAPS = Gst.Caps.from_string(
    'video/x-raw(memory:NVMM), format=RGBA; video/x-raw, format=RGBA'
)
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
            'Reference to last frame with its timestamp and resolution.',
            'Reference to last frame with its timestamp and resolution.',
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
        self._last_frame: Optional[LastFrameRef] = None

        self._time_overlay = TimestampOverlay()
        self._transfer: Dict[
            Mode, Callable[[int, int, Frame, Frame], Gst.FlowReturn]
        ] = {
            Mode.SCALE_TO_FIT: self.scale_to_fit,
            Mode.CROP_TO_FIT: self.crop_to_fit,
        }
        self._use_gpu = nvds_to_gpu_mat is not None
        self._width = 0
        self._height = 0

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

    def do_set_caps(self, in_caps: Gst.Caps, out_caps: Gst.Caps):
        """Parse frame resolution from caps."""

        self.logger.debug('Sink caps changed to %s', in_caps)
        struct: Gst.Structure = in_caps.get_structure(0)
        self._width = struct.get_int('width').value
        self._height = struct.get_int('height').value
        self.logger.info('Frame resolution is %sx%s', self._width, self._height)

        return True

    def do_transform_ip(self, buffer: Gst.Buffer):
        self.logger.trace('Output frame resolution is %sx%s', self._width, self._height)
        last_frame = self._last_frame.frame
        now = datetime.now()
        timestamp = (
            last_frame.timestamp
            if last_frame is not None
            else datetime.fromtimestamp(0)
        )
        delay = now - timestamp
        try:
            if self._width == 0 or self._height == 0:
                raise RuntimeError('Frame resolution is not set')

            if last_frame is None or delay > self._max_delay:
                self.logger.debug(
                    'No new data received from the input. Sending stub image with the timestamp.'
                )
                return self._overlay_timestamp(buffer, now)

            if self._width == last_frame.width and self._height == last_frame.height:
                return self._clone_frame_content(buffer, last_frame)

            return self._transform_frame_content(buffer, last_frame)

        except Exception as e:
            error = f'Failed to process frame with PTS {buffer.pts}: {e}'
            self.logger.exception(error)
            frame = inspect.currentframe()
            gst_post_stream_failed_error(self, frame, __file__, error)
            return Gst.FlowReturn.ERROR

    def _overlay_timestamp(self, buffer: Gst.Buffer, timestamp: datetime):
        if self._use_gpu:
            with nvds_to_gpu_mat(buffer, batch_id=0) as output_frame:
                self._time_overlay.overlay_timestamp(
                    self._width,
                    self._height,
                    output_frame,
                    timestamp,
                )
        else:
            with self._gst_buffer_to_np_array(buffer) as output_frame:
                self._time_overlay.overlay_timestamp(
                    self._width,
                    self._height,
                    output_frame,
                    timestamp,
                )

        return Gst.FlowReturn.OK

    def _clone_frame_content(self, buffer: Gst.Buffer, last_frame: LastFrame):
        if self._use_gpu:
            with nvds_to_gpu_mat(buffer, batch_id=0) as output_frame:
                last_frame.content.copyTo(output_frame)
        else:
            mem: Gst.Buffer = Gst.Buffer.new_wrapped(last_frame.content)
            buffer.replace_all_memory(mem.get_memory(0))

        return Gst.FlowReturn.OK

    def _transform_frame_content(self, buffer: Gst.Buffer, last_frame: LastFrame):
        if self._use_gpu:
            with nvds_to_gpu_mat(buffer, batch_id=0) as output_frame:
                return self._transfer[self._mode](
                    last_frame.width,
                    last_frame.height,
                    last_frame.content,
                    output_frame,
                )

        input_frame = np.ndarray(
            shape=(last_frame.height, last_frame.width, 4),
            dtype=np.uint8,
            buffer=last_frame.content,
        )
        with self._gst_buffer_to_np_array(buffer) as output_frame:
            return self._transfer[self._mode](
                last_frame.width,
                last_frame.height,
                input_frame,
                output_frame,
            )

    def scale_to_fit(
        self,
        in_width: int,
        in_height: int,
        input_frame: Frame,
        output_frame: Frame,
    ):
        out_width = self._width
        out_height = self._height
        in_aspect_ratio = in_width / in_height
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

        left = (out_width - target_width) // 2
        top = (out_height - target_height) // 2
        if isinstance(input_frame, cv2.cuda.GpuMat):
            output_frame.setTo((0, 0, 0, 0))
            target = cv2.cuda.GpuMat(
                output_frame,
                (left, top, target_width, target_height),
            )
            cv2.cuda.resize(input_frame, (target_width, target_height), target)

        else:
            output_frame.fill(0)
            target = output_frame[top : top + target_height, left : left + target_width]
            cv2.resize(input_frame, (target_width, target_height), target)

        return Gst.FlowReturn.OK

    def crop_to_fit(
        self,
        in_width: int,
        in_height: int,
        input_frame: Frame,
        output_frame: Frame,
    ):
        out_width = self._width
        out_height = self._height
        target_width = min(in_width, out_width)
        target_height = min(in_height, out_height)

        self.logger.debug(
            'Cropping input image from %sx%s to %sx%s',
            in_width,
            in_height,
            target_width,
            target_height,
        )

        in_left = (in_width - target_width) // 2
        in_top = (in_height - target_height) // 2
        out_left = (out_width - target_width) // 2
        out_top = (out_height - target_height) // 2
        if isinstance(input_frame, cv2.cuda.GpuMat):
            output_frame.setTo((0, 0, 0, 0))
            source = cv2.cuda.GpuMat(
                input_frame,
                (in_left, in_top, target_width, target_height),
            )
            target = cv2.cuda.GpuMat(
                output_frame,
                (out_left, out_top, target_width, target_height),
            )
            source.copyTo(target)

        else:
            output_frame.fill(0)
            output_frame[
                out_top : out_top + target_height,
                out_left : out_left + target_width,
            ] = input_frame[
                in_top : in_top + target_height,
                in_left : in_left + target_width,
            ]

        return Gst.FlowReturn.OK

    @contextmanager
    def _gst_buffer_to_np_array(self, buffer: Gst.Buffer):
        """Convert Gst.Buffer to a writable numpy array.
        All the changes will be applied to the buffer.
        """

        with map_gst_buffer(buffer, Gst.MapFlags.READ | Gst.MapFlags.WRITE) as map_info:
            np_arr = np.ctypeslib.as_array(
                map_info.data,
                shape=(self._height, self._width, 4),
            )
            yield np_arr


# register plugin
GObject.type_register(AlwaysOnRtspFrameProcessor)
__gstelementfactory__ = (
    AlwaysOnRtspFrameProcessor.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    AlwaysOnRtspFrameProcessor,
)
