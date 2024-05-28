from collections import deque
from fractions import Fraction
from typing import Any, Deque, Dict, Optional, Tuple

import pyds
from pygstsavantframemeta import gst_buffer_add_savant_frame_meta
from pynvbufsurfacegenerator import NvBufSurfaceGenerator
from savant_rs.pipeline2 import VideoPipeline
from savant_rs.primitives import VideoFrame, VideoFrameContent

from savant.api.constants import DEFAULT_TIME_BASE
from savant.config.schema import FrameParameters, PipelineElement
from savant.deepstream.buffer_processor import create_buffer_processor
from savant.deepstream.pipeline import NvDsPipeline
from savant.deepstream.source_output import SourceOutputH26X, create_source_output
from savant.gstreamer import Gst
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec
from savant.utils.logging import get_logger
from savant.utils.source_info import SourceInfoRegistry


class AuxiliaryStreamInternal:
    """Internal auxiliary stream implementation."""

    def __init__(
        self,
        source_id: str,
        sources: SourceInfoRegistry,
        width: int,
        height: int,
        framerate: str,
        codec_params: Dict[str, Any],
        video_pipeline: VideoPipeline,
        stage_name: str,
        gst_pipeline: NvDsPipeline,
        pad: Gst.Pad,
    ):
        self._logger = get_logger(f'{__name__}.{source_id}')
        self._source_id = source_id
        self._sources = sources
        self._source_info = sources.init_source(source_id)
        self._width = width
        self._height = height
        self._framerate = framerate
        framerate_fraction = Fraction(framerate)
        self._frame_duration = int(Gst.SECOND / framerate_fraction)
        self._codec_params = codec_params
        self._video_pipeline = video_pipeline
        self._stage_name = stage_name
        self._gst_pipeline = gst_pipeline
        self._pad = pad

        self._is_opened = False

        self._fakesink: Optional[Gst.Element] = None

        # None means EOS
        self._pending_buffers: Deque[Optional[Gst.Buffer]] = deque()

        codec = CODEC_BY_NAME[codec_params['codec']]
        if codec not in [Codec.H264, Codec.HEVC]:
            raise ValueError(f'Unsupported codec: {codec}')

        frame_params = FrameParameters(width=width, height=height)
        self._source_output = create_source_output(
            frame_params=frame_params,
            output_frame=codec_params,
            video_pipeline=video_pipeline,
            queue_properties={},
        )
        if not isinstance(self._source_output, SourceOutputH26X):
            raise ValueError('Unsupported codec')
        self._codec = self._source_output.codec.name
        self._buffer_processor = create_buffer_processor(
            queue=gst_pipeline._queue,
            sources=sources,
            frame_params=frame_params,
            source_output=self._source_output,
            video_pipeline=video_pipeline,
            pass_through_mode=False,
        )

        self._caps: Gst.Caps = Gst.Caps.from_string(
            ','.join(
                [
                    'video/x-raw(memory:NVMM)',
                    f'width={width}',
                    f'height={height}',
                    'format=RGBA',
                    f'framerate={framerate}',
                ]
            )
        )
        self._memory_type = int(pyds.NVBUF_MEM_DEFAULT)
        self._surface_generator = NvBufSurfaceGenerator(
            self._caps,
            0,  # TODO: get gpu_id
            self._memory_type,
        )

    def create_frame(
        self,
        pts: int,
        duration: Optional[int] = None,
    ) -> Tuple[VideoFrame, Gst.Buffer]:
        if not self._is_opened:
            self._open()

        self._logger.debug('Creating frame %s', pts)
        if duration is None:
            duration = self._frame_duration
        frame = VideoFrame(
            source_id=self._source_id,
            framerate=self._framerate,
            width=self._width,
            height=self._height,
            codec=self._codec,
            content=VideoFrameContent.none(),
            keyframe=True,
            pts=pts,
            dts=None,
            duration=duration,
            time_base=DEFAULT_TIME_BASE,
        )
        self._logger.debug('Creating buffer for frame %s', frame.pts)
        buffer: Gst.Buffer = Gst.Buffer.new()
        self._surface_generator.create_surface(buffer)
        buffer.pts = frame.pts if frame.pts is not None else Gst.CLOCK_TIME_NONE
        buffer.dts = frame.dts if frame.dts is not None else Gst.CLOCK_TIME_NONE
        buffer.duration = (
            frame.duration if frame.duration is not None else Gst.CLOCK_TIME_NONE
        )
        frame_idx = self._video_pipeline.add_frame(self._stage_name, frame)
        self._logger.debug(
            'Added frame %s to pipeline. Frame IDX: %s.',
            frame.pts,
            frame_idx,
        )
        gst_buffer_add_savant_frame_meta(buffer, frame_idx)

        self._pending_buffers.append(buffer)

        return frame, buffer

    def eos(self) -> bool:
        if not self._is_opened:
            self._logger.warning('Auxiliary stream is not opened')
            return False
        if self._pending_buffers:
            self.flush()
        self._logger.info('Sending EOS to auxiliary stream')
        return self._pad.push_event(Gst.Event.new_eos())

    def flush(self) -> Gst.FlowReturn:
        self._logger.debug('Flushing %s buffers', len(self._pending_buffers))
        ret = Gst.FlowReturn.OK
        while self._pending_buffers:
            buffer = self._pending_buffers.popleft()
            if not self._is_opened:
                self._open()
            self._logger.debug('Pushing buffer %s', buffer.pts)
            ret = self._pad.push(buffer)
            if ret not in [
                Gst.FlowReturn.OK,
                Gst.FlowReturn.FLUSHING,
                Gst.FlowReturn.EOS,
                Gst.FlowReturn.CUSTOM_SUCCESS,
                Gst.FlowReturn.CUSTOM_SUCCESS_1,
                Gst.FlowReturn.CUSTOM_SUCCESS_2,
            ]:
                self._logger.error('Failed to push buffer %s: %s', buffer.pts, ret)
                break

        return ret

    def _open(self):
        if self._is_opened:
            self._logger.warning('Auxiliary stream is already opened')
            return

        self._logger.info('Opening auxiliary stream')
        self._logger.debug('Creating capsfilter')
        capsfilter = self._gst_pipeline.add_element(
            PipelineElement(
                'capsfilter',
                properties={'caps': self._caps.to_string()},
            ),
            link=False,
        )
        self._source_info.after_demuxer.append(capsfilter)
        assert self._pad.link(capsfilter.get_static_pad('sink')) == Gst.PadLinkReturn.OK

        self._logger.debug('Creating source output')
        sink_pad = self._gst_pipeline._add_source_output(
            source_info=self._source_info,
            link_to_demuxer=False,
            source_output=self._source_output,
            buffer_processor=self._buffer_processor,
        )

        assert capsfilter.get_static_pad('src').link(sink_pad) == Gst.PadLinkReturn.OK
        capsfilter.sync_state_with_parent()
        self._is_opened = True

        stream: Gst.Stream = Gst.Stream.new(
            stream_id=None,
            caps=None,
            type=Gst.StreamType.VIDEO,
            flags=Gst.StreamFlags.NONE,
        )
        self._logger.debug(
            'Starting new stream in pad %s with stream id %s',
            self._pad.get_name(),
            stream.stream_id,
        )
        self._pad.push_event(Gst.Event.new_stream_start(stream.stream_id))

        segment: Gst.Segment = Gst.Segment.new()
        segment.init(Gst.Format.TIME)
        self._logger.debug('Starting new segment in pad %s', self._pad.get_name())
        self._pad.push_event(Gst.Event.new_segment(segment))

        self._logger.info('Auxiliary stream opened')

    def close(self):
        self._pad.get_parent().release_request_pad(self._pad)


class AuxiliaryStreamRegistry:
    """Registry for auxiliary streams."""

    def __init__(self):
        self._streams: Dict[str, AuxiliaryStreamInternal] = {}

    def add_stream(self, stream: AuxiliaryStreamInternal):
        self._streams[stream._source_id] = stream

    def get_stream(self, source_id: str) -> Optional[AuxiliaryStreamInternal]:
        return self._streams.get(source_id)

    def remove_stream(self, source_id: str):
        self._streams.pop(source_id, None)

    def flush(self):
        for stream in self._streams.values():
            stream.flush()


class AuxiliaryStream:
    """Auxiliary stream for sending frames directly to sink with a different source ID.

    Do not create instances of this class directly. Use `NvDsPyFuncPlugin.auxiliary_stream` instead.
    """

    def __init__(
        self,
        internal: AuxiliaryStreamInternal,
        registry: AuxiliaryStreamRegistry,
    ):
        self._internal = internal
        self._registry = registry

    def create_frame(
        self,
        pts: int,
        duration: Optional[int] = None,
    ) -> Tuple[VideoFrame, Gst.Buffer]:
        """Create a frame for the auxiliary stream.

        :param pts: Presentation timestamp of the frame.
        :param duration: Duration of the frame.
        :return: Tuple of the frame metadata and the buffer.
        """

        return self._internal.create_frame(pts, duration)

    def eos(self) -> bool:
        """Send EOS to the auxiliary stream."""

        return self._internal.eos()

    def __del__(self):
        """Remove the auxiliary stream."""

        self._internal.flush()
        self._internal.eos()
        self._internal.close()
        self._registry.remove_stream(self._internal._source_id)
