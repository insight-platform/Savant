"""Classes for adding output elements to a DeepStream pipeline."""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import pyds
from pygstsavantframemeta import add_convert_savant_frame_meta_pad_probe

from savant.config.schema import PipelineElement, FrameParameters
from savant.gstreamer import Gst  # noqa:F401
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec, CodecInfo
from savant.gstreamer.pipeline import GstPipeline
from savant.gstreamer.utils import link_pads
from savant.utils.platform import is_aarch64
from savant.utils.source_info import SourceInfo, Resolution


class SourceOutput(ABC):
    """Adds an output elements to a DeepStream pipeline."""

    def __init__(self):
        self._logger = logging.getLogger(
            f'{self.__class__.__module__}.{self.__class__.__name__}'
        )

    @abstractmethod
    def dest_resolution(self, source_info: SourceInfo) -> Resolution:
        pass

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

    def dest_resolution(self, source_info: SourceInfo) -> Resolution:
        return Resolution(
            width=source_info.src_resolution.width,
            height=source_info.src_resolution.height,
        )

    def add_output(
        self,
        pipeline: GstPipeline,
        source_info: SourceInfo,
        input_pad: Gst.Pad,
    ) -> Gst.Pad:
        self._logger.debug(
            'Do not add additional output elements since we output only frame metadata'
        )
        return input_pad


class SourceOutputWithFrame(SourceOutput):
    """Adds an output elements to a DeepStream pipeline.
    Output contains frames along with metadata.
    """

    def __init__(self, frame_params: FrameParameters, when_tagged: Optional[str]):
        super().__init__()
        self._frame_params = frame_params
        self._when_tagged = when_tagged

    def dest_resolution(self, source_info: SourceInfo) -> Resolution:
        width = source_info.src_resolution.width
        height = source_info.src_resolution.height
        if self._frame_params.padding and self._frame_params.padding.keep:
            width = width * self._frame_params.total_width // self._frame_params.width
            height = (
                height * self._frame_params.total_height // self._frame_params.height
            )
        return Resolution(width=width, height=height)

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
            if self._when_tagged
            else None
        )

        add_convert_savant_frame_meta_pad_probe(input_pad, False)
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
        source_info.after_demuxer.append(output_converter)
        output_converter.sync_state_with_parent()
        self._logger.debug(
            'Added converter for video frames (source_id=%s)',
            source_info.source_id,
        )

        self._add_transform_elems(pipeline, source_info)

        output_capsfilter = pipeline.add_element(PipelineElement('capsfilter'))
        output_caps = self._build_output_caps(source_info)
        output_capsfilter.set_property('caps', output_caps)
        source_info.after_demuxer.append(output_capsfilter)
        output_capsfilter.sync_state_with_parent()
        self._logger.debug(
            'Added capsfilter with caps %s (source_id=%s)',
            output_caps,
            source_info.source_id,
        )
        capsfilter_src_pad = output_capsfilter.get_static_pad('src')

        if src_pad_not_tagged is not None:
            return self._add_frame_tag_funnel(
                pipeline=pipeline,
                source_info=source_info,
                src_pad_tagged=capsfilter_src_pad,
                src_pad_not_tagged=src_pad_not_tagged,
            )

        return capsfilter_src_pad

    def _add_frame_tag_filter(
        self,
        pipeline: GstPipeline,
        source_info: SourceInfo,
    ) -> Gst.Pad:
        frame_tag_filter = pipeline.add_element(
            PipelineElement(
                'frame_tag_filter',
                properties={
                    'source-id': source_info.source_id,
                    'tag': self._when_tagged,
                },
            )
        )
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
        link_pads(
            frame_tag_filter.get_static_pad('src_tagged'),
            queue_tagged.get_static_pad('sink'),
        )

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
        queue_not_tagged = pipeline.add_element(
            PipelineElement('queue'),
            link=False,
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
            queue_not_tagged.get_static_pad('src'),
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
    def _build_output_caps(self, source_info: SourceInfo) -> Gst.Caps:
        pass


class SourceOutputRawRgba(SourceOutputWithFrame):
    """Adds an output elements to a DeepStream pipeline.
    Output contains raw-rgba frames along with metadata.
    """

    def __init__(self, frame_params: FrameParameters):
        super().__init__(frame_params, when_tagged=None)

    def _add_transform_elems(self, pipeline: GstPipeline, source_info: SourceInfo):
        pass

    def _build_output_caps(self, source_info: SourceInfo) -> Gst.Caps:
        return Gst.Caps.from_string(
            ', '.join(
                [
                    'video/x-raw(memory:NVMM)',
                    'format=RGBA',
                    f'width={source_info.dest_resolution.width}',
                    f'height={source_info.dest_resolution.height}',
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
        params: Optional[Dict[str, Any]],
        frame_params: FrameParameters,
        when_tagged: Optional[str],
    ):
        """
        :param codec: Codec for output frames.
        :param params: Parameters of the encoder.
        """

        super().__init__(frame_params=frame_params, when_tagged=when_tagged)
        self._codec = codec
        self._params = params or {}

    @property
    def codec(self) -> CodecInfo:
        return self._codec

    def _add_transform_elems(self, pipeline: GstPipeline, source_info: SourceInfo):
        encoder = pipeline.add_element(
            PipelineElement(self._codec.encoder, properties=self._params)
        )
        source_info.after_demuxer.append(encoder)
        encoder.sync_state_with_parent()
        self._logger.debug(
            'Added encoder %s with params %s', self._codec.encoder, self._params
        )

    def _build_output_caps(self, source_info: SourceInfo) -> Gst.Caps:
        return Gst.Caps.from_string(
            ', '.join(
                [
                    self._codec.caps_name,
                    f'width={source_info.dest_resolution.width}',
                    f'height={source_info.dest_resolution.height}',
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
        source_info.after_demuxer.append(parser)
        parser.sync_state_with_parent()
        self._logger.debug(
            'Added parser %s with params %s', self._codec.parser, parser_params
        )


class SourceOutputPng(SourceOutputEncoded):
    """Adds an output elements to a DeepStream pipeline.
    Output contains frames encoded with PNG codec along with metadata.
    """

    def dest_resolution(self, source_info: SourceInfo) -> Resolution:
        # Rounding resolution to the multiple of 8 to avoid cuda errors.
        dest_resolution = super().dest_resolution(source_info)
        return Resolution(
            width=round(dest_resolution.width / 8) * 8,
            height=round(dest_resolution.height / 8) * 8,
        )


def create_source_output(
    frame_params: FrameParameters,
    output_frame: Optional[Dict[str, Any]],
) -> SourceOutput:
    if not output_frame:
        return SourceOutputOnlyMeta()

    codec = CODEC_BY_NAME[output_frame['codec']]
    if codec == Codec.RAW_RGBA:
        return SourceOutputRawRgba(frame_params=frame_params)

    encoder_params = output_frame.get('encoder_params', {})
    when_tagged = output_frame.get('when_tagged')
    if codec in [Codec.H264, Codec.HEVC]:
        return SourceOutputH26X(
            codec=codec.value,
            params=encoder_params,
            frame_params=frame_params,
            when_tagged=when_tagged,
        )

    if codec == Codec.PNG:
        return SourceOutputPng(
            codec=codec.value,
            params=encoder_params,
            frame_params=frame_params,
            when_tagged=when_tagged,
        )

    return SourceOutputEncoded(
        codec=codec,
        params=encoder_params,
        frame_params=frame_params,
        when_tagged=when_tagged,
    )
