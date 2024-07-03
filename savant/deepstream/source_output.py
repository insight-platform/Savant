"""Classes for adding output elements to a DeepStream pipeline."""

from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, Dict, Optional

import pyds
from pygstsavantframemeta import (
    add_pad_probe_to_move_frame,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)
from savant_rs.pipeline2 import VideoPipeline
from savant_rs.primitives import VideoFrameContent

from savant.api.constants import DEFAULT_NAMESPACE
from savant.config.schema import (
    FrameParameters,
    FrameProcessingCondition,
    PipelineElement,
)
from savant.deepstream.utils.iterator import nvds_frame_meta_iterator
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

    def remove_output(self, pipeline: Gst.Pipeline, source_info: SourceInfo):
        """Remove output elements from the pipeline.

        :param pipeline: Target pipeline.
        :param source_info: Video source information.
        """

        for elem in source_info.source_output_elements:
            self._logger.debug('Removing element %s', elem.get_name())
            elem.set_locked_state(True)
            elem.set_state(Gst.State.NULL)
            pipeline.remove(elem)
        source_info.source_output_elements = []

    @property
    @abstractmethod
    def codec(self) -> Optional[CodecInfo]:
        pass


class SourceOutputOnlyMeta(SourceOutput):
    """Adds an output elements to a DeepStream pipeline.
    Output contains only frames metadata (without the frames).
    """

    def __init__(
        self,
        video_pipeline: VideoPipeline,
        condition: Optional[FrameProcessingCondition] = None,
    ):
        super().__init__(video_pipeline)
        self._condition = condition

    def add_output(
        self,
        pipeline: GstPipeline,
        source_info: SourceInfo,
        input_pad: Gst.Pad,
    ) -> Gst.Pad:
        self._logger.debug(
            'Do not add additional output elements since we output only frame metadata'
        )
        if self._condition is not None and self._condition.tag is not None:
            input_pad.add_probe(Gst.PadProbeType.BUFFER, self._check_encoding_condition)
        add_pad_probe_to_move_frame(input_pad, self._video_pipeline, 'sink')
        return input_pad

    @property
    def codec(self) -> Optional[CodecInfo]:
        return None

    def _check_encoding_condition(self, pad: Gst.Pad, info: Gst.PadProbeInfo):
        buffer: Gst.Buffer = info.get_buffer()
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            frame_id = savant_frame_meta.idx
            frame, _ = self._video_pipeline.get_independent_frame(frame_id)
            attr = frame.get_attribute(DEFAULT_NAMESPACE, self._condition.tag)
            if attr is not None:
                self._logger.debug(
                    'Frame %s has tag %r. Keeping content.',
                    frame_id,
                    self._condition.tag,
                )
            else:
                self._logger.debug(
                    'Frame %s does not have tag %r. Removing content.',
                    frame_id,
                    self._condition.tag,
                )
                frame.content = VideoFrameContent.none()

        return Gst.PadProbeReturn.OK


class SourceOutputWithFrame(SourceOutput):
    """Adds an output elements to a DeepStream pipeline.
    Output contains frames along with metadata.
    """

    def __init__(
        self,
        codec: CodecInfo,
        frame_params: FrameParameters,
        condition: FrameProcessingCondition,
        video_pipeline: VideoPipeline,
        queue_properties: Dict[str, int],
    ):
        super().__init__(video_pipeline)
        self._codec = codec
        self._frame_params = frame_params
        self._condition = condition
        self._queue_properties = queue_properties

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
        source_info.source_output_elements.append(output_converter)
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
        source_info.source_output_elements.append(output_capsfilter)
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

    @property
    def codec(self) -> Optional[CodecInfo]:
        return self._codec

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

        queue_tagged = pipeline.add_element(
            PipelineElement('queue', properties=self._queue_properties), link=False
        )
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
            PipelineElement('queue', properties=self._queue_properties),
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


class SourceOutputRaw(SourceOutputWithFrame):
    """Adds an output elements to a DeepStream pipeline.
    Output contains raw frames along with metadata.
    """

    def __init__(
        self,
        codec: CodecInfo,
        frame_params: FrameParameters,
        video_pipeline: VideoPipeline,
        queue_properties: Dict[str, int],
    ):
        super().__init__(
            codec=codec,
            frame_params=frame_params,
            condition=FrameProcessingCondition(),
            video_pipeline=video_pipeline,
            queue_properties=queue_properties,
        )

    def _add_transform_elems(self, pipeline: GstPipeline, source_info: SourceInfo):
        pass

    def _build_output_caps(self, width: int, height: int) -> Gst.Caps:
        if self.codec.name == Codec.RAW_RGBA.value.name:
            codec_caps = [
                'video/x-raw(memory:NVMM)',
                'format=RGBA',
            ]
        else:
            codec_caps = [self.codec.caps_with_params]

        return Gst.Caps.from_string(
            ', '.join(
                codec_caps
                + [
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
        queue_properties: Dict[str, int],
    ):
        """
        :param codec: Codec for output frames.
        """

        super().__init__(
            codec=codec,
            frame_params=frame_params,
            condition=condition,
            video_pipeline=video_pipeline,
            queue_properties=queue_properties,
        )
        self._output_frame = output_frame
        self._encoder = self._codec.encoder(output_frame.get('encoder'))
        self._params = output_frame.get('encoder_params') or {}

    @property
    def encoder(self) -> str:
        return self._encoder

    def _add_transform_elems(self, pipeline: GstPipeline, source_info: SourceInfo):
        encoder = self._create_encoder(pipeline)
        source_info.source_output_elements.append(encoder)
        encoder.sync_state_with_parent()
        self._logger.debug(
            'Added encoder %s with params %s', self._encoder, self._params
        )

    def _create_encoder(self, pipeline: GstPipeline):
        encoder = pipeline.add_element(
            PipelineElement(self._encoder, properties=self._params)
        )
        add_pad_probe_to_move_frame(
            encoder.get_static_pad('sink'),
            self._video_pipeline,
            'encode',
        )

        return encoder

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
        source_info.source_output_elements.append(parser)
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


class SourceOutputNvJpeg(SourceOutputEncoded):
    """Adds an output elements to a DeepStream pipeline.
    Output contains frames encoded with jpeg codec along with metadata.

    Separate handling for nvjpegenc element to reuse it needed as a workaround for bug
    https://forums.developer.nvidia.com/t/nvjpegenc-dont-release-gpu-memory-when-gst-element-removed-from-pipeline
    """

    def __init__(
        self,
        codec: CodecInfo,
        output_frame: Dict[str, Any],
        frame_params: FrameParameters,
        condition: FrameProcessingCondition,
        video_pipeline: VideoPipeline,
        queue_properties: Dict[str, int],
    ):
        super().__init__(
            codec=codec,
            output_frame=output_frame,
            frame_params=frame_params,
            condition=condition,
            video_pipeline=video_pipeline,
            queue_properties=queue_properties,
        )
        # Pool of nvjpegenc elements to reuse them
        self._encoder_pool: Deque[Gst.Element] = deque()
        self._use_nvenc = codec.nv_encoder == self._encoder

    def _create_encoder(self, pipeline: GstPipeline):
        if self._use_nvenc and self._encoder_pool:
            self._logger.debug(
                'Reusing nvjpegenc element from the pool. Pool size: %s.',
                len(self._encoder_pool),
            )
            encoder = self._encoder_pool.popleft()
            encoder.set_locked_state(False)
            pipeline.link_element(encoder)
        else:
            encoder = super()._create_encoder(pipeline)

        return encoder

    def remove_output(self, pipeline: Gst.Pipeline, source_info: SourceInfo):
        """Remove output elements from the pipeline.

        :param pipeline: Target pipeline.
        :param source_info: Video source information.
        """

        if not self._use_nvenc:
            return super().remove_output(pipeline, source_info)

        for elem in source_info.source_output_elements:
            if elem.get_factory().get_name() == 'nvjpegenc':
                self._logger.debug('Adding element %s to the pool', elem.get_name())
                self._encoder_pool.append(elem)
                elem.set_locked_state(True)
                elem.set_state(Gst.State.NULL)
                self._logger.debug('Encoder pool size: %s', len(self._encoder_pool))
                continue

            self._logger.debug('Removing element %s', elem.get_name())
            elem.set_locked_state(True)
            elem.set_state(Gst.State.NULL)
            pipeline.remove(elem)
        source_info.source_output_elements = []


def create_source_output(
    frame_params: FrameParameters,
    output_frame: Optional[Dict[str, Any]],
    video_pipeline: VideoPipeline,
    queue_properties: Dict[str, int],
) -> SourceOutput:
    """Create an instance of SourceOutput class based on the output_frame config."""

    if not output_frame:
        return SourceOutputOnlyMeta(video_pipeline=video_pipeline)

    condition = FrameProcessingCondition(**(output_frame.get('condition') or {}))
    if output_frame['codec'] == 'copy':
        return SourceOutputOnlyMeta(
            video_pipeline=video_pipeline,
            condition=condition,
        )

    codec = CODEC_BY_NAME[output_frame['codec']]
    if codec.value.is_raw:
        return SourceOutputRaw(
            codec=codec.value,
            frame_params=frame_params,
            video_pipeline=video_pipeline,
            queue_properties=queue_properties,
        )

    condition = FrameProcessingCondition(**(output_frame.get('condition') or {}))
    if codec in [Codec.H264, Codec.HEVC]:
        return SourceOutputH26X(
            codec=codec.value,
            output_frame=output_frame,
            frame_params=frame_params,
            condition=condition,
            video_pipeline=video_pipeline,
            queue_properties=queue_properties,
        )

    if codec == Codec.JPEG and not is_aarch64():
        return SourceOutputNvJpeg(
            codec=codec.value,
            output_frame=output_frame,
            frame_params=frame_params,
            condition=condition,
            video_pipeline=video_pipeline,
            queue_properties=queue_properties,
        )

    return SourceOutputEncoded(
        codec=codec.value,
        output_frame=output_frame,
        frame_params=frame_params,
        condition=condition,
        video_pipeline=video_pipeline,
        queue_properties=queue_properties,
    )
