"""Classes for adding output elements to a DeepStream pipeline."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pyds
from pygstsavantframemeta import add_pad_probe_to_move_frame
from savant_rs.pipeline2 import VideoPipeline

from savant.config.schema import (
    FrameParameters,
    FrameProcessingCondition,
    PipelineElement,
)
from savant.gstreamer import Gst  # noqa:F401
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec, CodecInfo
from savant.gstreamer.pipeline import GstPipeline
from savant.gstreamer.utils import link_pads
from savant.utils.logging import get_logger
from savant.utils.platform import is_aarch64
from savant.utils.source_info import SourceInfo


class SourceOutput(ABC):
    """Adds an output elements to a DeepStream pipeline."""

    def __init__(self, video_pipeline: VideoPipeline):
        self._logger = get_logger(
            f'{self.__class__.__module__}.{self.__class__.__name__}'
        )
        self._video_pipeline = video_pipeline

    @abstractmethod
    def add_output(
        self,
        pipeline: GstPipeline,
        source_info: SourceInfo,
        input_pad: Gst.Pad,
    ) -> Gst.Pad:
        """Add an output elements to the pipeline.

        :param pipeline: Target pipeline.
        :param source_info: Video source information.
        :param input_pad: A sink pad to which elements must be linked.
        :returns Src pad that will be connected to fakesink element.
        """


class SourceOutputOnlyMeta(SourceOutput):
    """Adds an output elements to a DeepStream pipeline.
    Output contains only frames metadata (without the frames).
    """

    def add_output(
        self,
        pipeline: GstPipeline,
        source_info: SourceInfo,
        input_pad: Gst.Pad,
    ) -> Gst.Pad:
        self._logger.debug(
            'Do not add additional output elements since we output only frame metadata'
        )
        add_pad_probe_to_move_frame(input_pad, self._video_pipeline, 'sink')
        return input_pad


class SourceOutputWithFrame(SourceOutput):
    """Adds an output elements to a DeepStream pipeline.
    Output contains frames along with metadata.
    """

    def __init__(
        self,
        frame_params: FrameParameters,
        condition: FrameProcessingCondition,
        video_pipeline: VideoPipeline,
    ):
        super().__init__(video_pipeline)
        self._frame_params = frame_params
        self._condition = condition

    def add_output(
        self,
        pipeline: GstPipeline,
        source_info: SourceInfo,
        input_pad: Gst.Pad,
    ) -> Gst.Pad:
        self._logger.debug(
            'Adding additional output elements (source_id=%s)',
            source_info.source_id,
        )

        src_pad_not_tagged = (
            self._add_frame_tag_filter(pipeline, source_info)
            if self._condition.tag
            else None
        )

        self._logger.debug(
            'Added pad probe to convert savant frame meta from NvDsMeta to GstMeta (source_id=%s)',
            source_info.source_id,
        )
        output_converter_props = {}
        if not is_aarch64():
            output_converter_props['nvbuf-memory-type'] = int(
                pyds.NVBUF_MEM_CUDA_UNIFIED
            )
        if self._frame_params.padding and not self._frame_params.padding.keep:
            output_converter_props['src_crop'] = ':'.join(
                str(x)
                for x in [
                    self._frame_params.padding.left,
                    self._frame_params.padding.top,
                    self._frame_params.width,
                    self._frame_params.height,
                ]
            )
        self._logger.debug(
            'Output converter properties: %s (source_id=%s)',
            output_converter_props,
            source_info.source_id,
        )
        output_converter = pipeline.add_element(
            PipelineElement(
                'nvvideoconvert',
                properties=output_converter_props,
            ),
        )
        output_converter_sink_pad: Gst.Pad = output_converter.get_static_pad('sink')
        add_pad_probe_to_move_frame(
            output_converter_sink_pad,
            self._video_pipeline,
            'sink-convert',
        )
        source_info.after_demuxer.append(output_converter)
        output_converter.sync_state_with_parent()
        self._logger.debug(
            'Added converter for video frames (source_id=%s)',
            source_info.source_id,
        )

        self._add_transform_elems(pipeline, source_info)

        output_capsfilter = pipeline.add_element(PipelineElement('capsfilter'))
        output_caps = self._build_output_caps(
            self._frame_params.output_width,
            self._frame_params.output_height,
        )
        output_capsfilter.set_property('caps', output_caps)
        source_info.after_demuxer.append(output_capsfilter)
        output_capsfilter.sync_state_with_parent()
        self._logger.debug(
            'Added capsfilter with caps %s (source_id=%s)',
            output_caps,
            source_info.source_id,
        )
        add_pad_probe_to_move_frame(
            output_capsfilter.get_static_pad('sink'),
            self._video_pipeline,
            'sink-capsfilter',
        )
        src_pad = output_capsfilter.get_static_pad('src')

        if src_pad_not_tagged is not None:
            src_pad = self._add_frame_tag_funnel(
                pipeline=pipeline,
                source_info=source_info,
                src_pad_tagged=src_pad,
                src_pad_not_tagged=src_pad_not_tagged,
            )

        add_pad_probe_to_move_frame(src_pad, self._video_pipeline, 'sink')

        return src_pad

    def _add_frame_tag_filter(
        self,
        pipeline: GstPipeline,
        source_info: SourceInfo,
    ) -> Gst.Pad:
        """Add frame_tag_filter element to the pipeline.

        :returns: src pad for not tagged frames.
        """

        frame_tag_filter = pipeline.add_element(
            PipelineElement(
                'frame_tag_filter',
                properties={
                    'source-id': source_info.source_id,
                    'tag': self._condition.tag,
                },
            )
        )
        add_pad_probe_to_move_frame(
            frame_tag_filter.get_static_pad('sink'),
            self._video_pipeline,
            'frame-tag-filter',
        )
        frame_tag_filter.set_property('pipeline', self._video_pipeline)
        self._logger.debug(
            'Added frame_tag_filter for video frames (source_id=%s)',
            source_info.source_id,
        )
        src_pad_not_tagged = frame_tag_filter.get_static_pad('src_not_tagged')

        queue_tagged = pipeline.add_element(PipelineElement('queue'), link=False)
        self._logger.debug(
            'Added queue for tagged video frames (source_id=%s)',
            source_info.source_id,
        )
        src_pad_tagged: Gst.Pad = frame_tag_filter.get_static_pad('src_tagged')
        add_pad_probe_to_move_frame(
            src_pad_tagged,
            self._video_pipeline,
            'queue-tagged',
        )
        link_pads(src_pad_tagged, queue_tagged.get_static_pad('sink'))

        queue_tagged.sync_state_with_parent()
        frame_tag_filter.sync_state_with_parent()

        return src_pad_not_tagged

    def _add_frame_tag_funnel(
        self,
        pipeline: GstPipeline,
        source_info: SourceInfo,
        src_pad_tagged: Gst.Pad,
        src_pad_not_tagged: Gst.Pad,
    ) -> Gst.Pad:
        """Add frame_tag_funnel element to the pipeline.

        :returns: src pad of the frame_tag_funnel element.
        """

        add_pad_probe_to_move_frame(
            src_pad_tagged,
            self._video_pipeline,
            'frame-tag-funnel',
        )
        add_pad_probe_to_move_frame(
            src_pad_not_tagged,
            self._video_pipeline,
            'queue-not-tagged',
        )
        queue_not_tagged = pipeline.add_element(
            PipelineElement('queue'),
            link=False,
        )
        queue_not_tagged_src_pad: Gst.Pad = queue_not_tagged.get_static_pad('src')
        add_pad_probe_to_move_frame(
            queue_not_tagged_src_pad,
            self._video_pipeline,
            'frame-tag-funnel',
        )
        self._logger.debug(
            'Added queue for not tagged video frames (source_id=%s)',
            source_info.source_id,
        )
        link_pads(src_pad_not_tagged, queue_not_tagged.get_static_pad('sink'))

        frame_tag_funnel = pipeline.add_element(
            PipelineElement('frame_tag_funnel'),
            link=False,
        )
        self._logger.debug(
            'Added frame_tag_funnel for video frames (source_id=%s)',
            source_info.source_id,
        )
        link_pads(
            queue_not_tagged_src_pad,
            frame_tag_funnel.get_static_pad('sink_not_tagged'),
        )
        link_pads(src_pad_tagged, frame_tag_funnel.get_static_pad('sink_tagged'))

        frame_tag_funnel.sync_state_with_parent()
        queue_not_tagged.sync_state_with_parent()

        return frame_tag_funnel.get_static_pad('src')

    @abstractmethod
    def _add_transform_elems(self, pipeline: GstPipeline, source_info: SourceInfo):
        pass

    @abstractmethod
    def _build_output_caps(self, width: int, height: int) -> Gst.Caps:
        pass


class SourceOutputRawRgba(SourceOutputWithFrame):
    """Adds an output elements to a DeepStream pipeline.
    Output contains raw-rgba frames along with metadata.
    """

    def __init__(self, frame_params: FrameParameters, video_pipeline: VideoPipeline):
        super().__init__(
            frame_params,
            condition=FrameProcessingCondition(),
            video_pipeline=video_pipeline,
        )

    def _add_transform_elems(self, pipeline: GstPipeline, source_info: SourceInfo):
        pass

    def _build_output_caps(self, width: int, height: int) -> Gst.Caps:
        return Gst.Caps.from_string(
            ', '.join(
                [
                    'video/x-raw(memory:NVMM)',
                    'format=RGBA',
                    f'width={width}',
                    f'height={height}',
                ]
            )
        )


class SourceOutputEncoded(SourceOutputWithFrame):
    """Adds an output elements to a DeepStream pipeline.
    Output contains encoded frames along with metadata.
    """

    def __init__(
        self,
        codec: CodecInfo,
        output_frame: Dict[str, Any],
        frame_params: FrameParameters,
        condition: FrameProcessingCondition,
        video_pipeline: VideoPipeline,
    ):
        """
        :param codec: Codec for output frames.
        """

        super().__init__(
            frame_params=frame_params,
            condition=condition,
            video_pipeline=video_pipeline,
        )
        self._codec = codec
        self._output_frame = output_frame
        self._encoder = self._codec.encoder(output_frame.get('encoder'))
        self._params = output_frame.get('encoder_params') or {}

    @property
    def codec(self) -> CodecInfo:
        return self._codec

    @property
    def encoder(self) -> str:
        return self._encoder

    def _add_transform_elems(self, pipeline: GstPipeline, source_info: SourceInfo):
        encoder = pipeline.add_element(
            PipelineElement(self._encoder, properties=self._params)
        )
        add_pad_probe_to_move_frame(
            encoder.get_static_pad('sink'),
            self._video_pipeline,
            'encode',
        )
        source_info.after_demuxer.append(encoder)
        encoder.sync_state_with_parent()
        self._logger.debug(
            'Added encoder %s with params %s', self._encoder, self._params
        )

    def _build_output_caps(self, width: int, height: int) -> Gst.Caps:
        return Gst.Caps.from_string(
            ', '.join(
                [
                    self._codec.caps_with_params,
                    f'width={width}',
                    f'height={height}',
                ]
            )
        )


class SourceOutputH26X(SourceOutputEncoded):
    """Adds an output elements to a DeepStream pipeline.
    Output contains frames encoded with h264 or h265 (hevc) codecs along with metadata.
    """

    def _add_transform_elems(self, pipeline: GstPipeline, source_info: SourceInfo):
        super()._add_transform_elems(pipeline, source_info)
        # A parser for codecs h264, h265 is added to include
        # the Sequence Parameter Set (SPS) and the Picture Parameter Set (PPS)
        # to IDR frames in the video stream. SPS and PPS are needed
        # to correct recording or playback not from the beginning
        # of the video stream.
        parser_params = {'config-interval': -1}
        parser = pipeline.add_element(
            PipelineElement(self._codec.parser, properties=parser_params)
        )
        add_pad_probe_to_move_frame(
            parser.get_static_pad('sink'),
            self._video_pipeline,
            'parse',
        )
        source_info.after_demuxer.append(parser)
        parser.sync_state_with_parent()
        self._logger.debug(
            'Added parser %s with params %s', self._codec.parser, parser_params
        )

    def _build_output_caps(self, width: int, height: int) -> Gst.Caps:
        caps_params = [
            self._codec.caps_with_params,
            f'width={width}',
            f'height={height}',
        ]
        if (
            self._codec.name == Codec.H264.value.name
            and self._encoder == self._codec.sw_encoder
        ):
            profile = self._output_frame.get('profile')
            if profile is None:
                profile = 'baseline'
            caps_params.append(f'profile={profile}')

        return Gst.Caps.from_string(', '.join(caps_params))


def create_source_output(
    frame_params: FrameParameters,
    output_frame: Optional[Dict[str, Any]],
    video_pipeline: VideoPipeline,
) -> SourceOutput:
    """Create an instance of SourceOutput class based on the output_frame config."""

    if not output_frame:
        return SourceOutputOnlyMeta(video_pipeline=video_pipeline)

    codec = CODEC_BY_NAME[output_frame['codec']]
    if codec == Codec.RAW_RGBA:
        return SourceOutputRawRgba(
            frame_params=frame_params,
            video_pipeline=video_pipeline,
        )

    condition = FrameProcessingCondition(**(output_frame.get('condition') or {}))
    if codec in [Codec.H264, Codec.HEVC]:
        return SourceOutputH26X(
            codec=codec.value,
            output_frame=output_frame,
            frame_params=frame_params,
            condition=condition,
            video_pipeline=video_pipeline,
        )

    return SourceOutputEncoded(
        codec=codec.value,
        output_frame=output_frame,
        frame_params=frame_params,
        condition=condition,
        video_pipeline=video_pipeline,
    )
