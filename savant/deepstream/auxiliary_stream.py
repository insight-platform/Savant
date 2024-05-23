from collections import deque
from fractions import Fraction
from typing import Any, Deque, Dict, Optional, Tuple

import pyds
from pygstsavantframemeta import NvBufSurfaceGenerator, gst_buffer_add_savant_frame_meta
from savant_rs.pipeline2 import VideoPipeline
from savant_rs.primitives import VideoFrame, VideoFrameContent

from savant.api.constants import DEFAULT_TIME_BASE
from savant.config.schema import FrameParameters, PipelineElement
from savant.deepstream.buffer_processor import create_buffer_processor
from savant.deepstream.pipeline import NvDsPipeline
from savant.deepstream.source_output import SourceOutputH26X, create_source_output
from savant.gstreamer import Gst, GstApp
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec
from savant.utils.logging import get_logger
from savant.utils.source_info import SourceInfoRegistry


class AuxiliaryStream:
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

        self._is_opened = False

        self._appsrc: Optional[GstApp.AppSrc] = None
        self._fakesink: Optional[Gst.Element] = None
        self._pending_buffers: Deque[Gst.Buffer] = deque()

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

    def eos(self) -> Gst.FlowReturn:
        self._logger.info('Sending EOS to auxiliary stream')
        return self._appsrc.emit('end-of-stream')

    def _flush(self) -> Gst.FlowReturn:
        self._logger.debug('Flushing %s buffers', len(self._pending_buffers))
        ret = Gst.FlowReturn.OK
        while self._pending_buffers:
            buffer = self._pending_buffers.popleft()
            self._logger.debug('Pushing buffer %s', buffer.pts)
            ret = self._appsrc.emit('push-buffer', buffer)
            if ret != Gst.FlowReturn.OK:
                self._logger.error('Failed to push buffer %s: %s', ret)
                break

        return ret

    def _open(self):
        if self._is_opened:
            self._logger.warning('Auxiliary stream is already opened')
            return

        self._logger.info('Opening auxiliary stream')
        # TODO: use pad from pyfunc instead of fakesink
        self._logger.debug('Creating appsrc')
        self._appsrc = self._gst_pipeline.add_element(
            PipelineElement(
                'appsrc',
                properties={
                    'emit-signals': False,
                    'format': int(Gst.Format.TIME),
                    'max-buffers': 1,
                    'block': True,
                },
            ),
            link=False,
        )
        self._source_info.after_demuxer.append(self._appsrc)

        self._logger.debug('Creating capsfilter')
        capsfilter = self._gst_pipeline.add_element(
            PipelineElement(
                'capsfilter',
                properties={'caps': self._caps.to_string()},
            ),
            link=False,
        )
        self._source_info.after_demuxer.append(capsfilter)
        assert self._appsrc.link(capsfilter)

        self._logger.debug('Creating source output')
        sink_pad = self._gst_pipeline._add_source_output(
            source_info=self._source_info,
            link_to_demuxer=False,
            source_output=self._source_output,
            buffer_processor=self._buffer_processor,
        )

        assert capsfilter.get_static_pad('src').link(sink_pad) == Gst.PadLinkReturn.OK
        capsfilter.sync_state_with_parent()
        self._appsrc.sync_state_with_parent()
        self._is_opened = True
        self._logger.info('Auxiliary stream opened')

    def _close(self):
        pass

    def __del__(self):
        if self._is_opened:
            self._close()
