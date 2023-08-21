"""SavantRsAddFrames element."""
import inspect
from typing import Any, NamedTuple, Optional

from pygstsavantframemeta import gst_buffer_add_savant_frame_meta
from savant_rs.pipeline2 import VideoPipeline
from savant_rs.primitives import VideoFrame, VideoFrameContent

from savant.api.constants import DEFAULT_FRAMERATE, DEFAULT_TIME_BASE
from savant.gstreamer import GObject, Gst, GstBase  # noqa: F401
from savant.gstreamer.utils import (
    RequiredPropertyError,
    propagate_gst_setting_error,
    required_property,
)
from savant.utils.logging import LoggerMixin


class FrameParams(NamedTuple):
    """Frame parameters."""

    width: int
    height: int
    framerate: str


class SavantRsAddFrames(LoggerMixin, GstBase.BaseTransform):
    """Adds Savant-rs VideoFrame to VideoPipeline."""

    GST_PLUGIN_NAME: str = 'savant_rs_add_frames'

    __gstmetadata__ = (
        'Savant-rs VideoFrame adder',
        'Transform',
        'Adds Savant-rs VideoFrame to VideoPipeline.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'sink',
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        ),
        Gst.PadTemplate.new(
            'src',
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        ),
    )

    __gproperties__ = {
        'source-id': (
            str,
            'Source ID.',
            'Set this source ID to video frames.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'pipeline': (
            object,
            'VideoPipeline object from savant-rs.',
            'VideoPipeline object from savant-rs.',
            GObject.ParamFlags.READWRITE,
        ),
        'pipeline-stage-name': (
            str,
            'Name of the pipeline stage.',
            'Name of the pipeline stage.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        # properties
        self._source_id: Optional[str] = None
        self._video_pipeline: Optional[VideoPipeline] = None
        self._pipeline_stage_name: Optional[str] = None
        # will be set after caps negotiation
        self._frame_params: Optional[FrameParams] = None

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        if prop.name == 'source-id':
            return self._source_id
        if prop.name == 'pipeline':
            return self._video_pipeline
        if prop.name == 'pipeline-stage-name':
            return self._pipeline_stage_name
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        if prop.name == 'source-id':
            self._source_id = value
        elif prop.name == 'pipeline':
            self._video_pipeline = value
        elif prop.name == 'pipeline-stage-name':
            self._pipeline_stage_name = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        """Do on plugin start."""

        try:
            required_property('source-id', self._source_id)
            required_property('pipeline', self._video_pipeline)
            required_property('pipeline-stage-name', self._pipeline_stage_name)
        except RequiredPropertyError as exc:
            self.logger.exception('Failed to start element: %s', exc, exc_info=True)
            frame = inspect.currentframe()
            propagate_gst_setting_error(self, frame, __file__, text=exc.args[0])
            return False

        return True

    def do_set_caps(  # pylint: disable=unused-argument
        self,
        in_caps: Gst.Caps,
        out_caps: Gst.Caps,
    ):
        """Checks caps after negotiations."""
        self.logger.info('Sink caps changed to %s', in_caps)
        struct: Gst.Structure = in_caps.get_structure(0)
        frame_width = struct.get_int('width').value
        frame_height = struct.get_int('height').value
        if struct.has_field('framerate'):
            _, framerate_num, framerate_demon = struct.get_fraction('framerate')
            framerate = f'{framerate_num}/{framerate_demon}'
        else:
            framerate = DEFAULT_FRAMERATE
        self._frame_params = FrameParams(
            width=frame_width,
            height=frame_height,
            framerate=framerate,
        )
        return True

    def do_transform_ip(self, buffer: Gst.Buffer):
        """Transform buffer in-place function."""
        self._logger.debug(
            'Adding frame to %s in buffer with PTS %s. is_all_memory_writable=%s',
            self._pipeline_stage_name,
            buffer.pts,
            buffer.is_all_memory_writable(),
        )
        keyframe = not buffer.has_flags(Gst.BufferFlags.DELTA_UNIT)
        video_frame = VideoFrame(
            source_id=self._source_id,
            framerate=self._frame_params.framerate,
            width=self._frame_params.width,
            height=self._frame_params.height,
            content=VideoFrameContent.none(),
            keyframe=keyframe,
            pts=buffer.pts,
            dts=buffer.dts if buffer.dts != Gst.CLOCK_TIME_NONE else None,
            duration=(
                buffer.duration if buffer.duration != Gst.CLOCK_TIME_NONE else None
            ),
            time_base=DEFAULT_TIME_BASE,
        )
        frame_id = self._video_pipeline.add_frame(
            self._pipeline_stage_name, video_frame
        )
        self._logger.debug(
            'Adding frame to %s in buffer with PTS %s. Frame ID: %s. is_all_memory_writable=%s',
            self._pipeline_stage_name,
            buffer.pts,
            frame_id,
            buffer.is_all_memory_writable(),
        )
        gst_buffer_add_savant_frame_meta(buffer, frame_id)
        self._logger.debug(
            'Adding frame to %s in buffer with PTS %s. Frame ID: %s. is_all_memory_writable=%s',
            self._pipeline_stage_name,
            buffer.pts,
            frame_id,
            buffer.is_all_memory_writable(),
        )
        return Gst.FlowReturn.OK


# register plugin
GObject.type_register(SavantRsAddFrames)
__gstelementfactory__ = (
    SavantRsAddFrames.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SavantRsAddFrames,
)
