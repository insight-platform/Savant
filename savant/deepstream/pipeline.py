"""DeepStream pipeline."""
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional, Union
import time
import numpy as np
import pyds

from pysavantboost import ObjectsPreprocessing
from pygstsavantframemeta import (
    add_convert_savant_frame_meta_pad_probe,
    gst_buffer_get_savant_frame_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)

from savant.converter.scale import scale_rbbox
from savant.deepstream.nvinfer.model import (
    NvInferRotatedObjectDetector,
    NvInferDetector,
    NvInferAttributeModel,
)
from savant.gstreamer import Gst, GLib  # noqa:F401
from savant.gstreamer.codecs import Codec, CODEC_BY_NAME
from savant.gstreamer.pipeline import GstPipeline
from savant.deepstream.metadata import (
    nvds_obj_meta_output_converter,
    nvds_attr_meta_output_converter,
)
from savant.gstreamer.metadata import (
    metadata_add_frame_meta,
    metadata_get_frame_meta,
    metadata_pop_frame_meta,
)
from savant.gstreamer.utils import on_pad_event, pad_to_source_id
from savant.deepstream.utils import (
    gst_nvevent_new_stream_eos,
    GST_NVEVENT_PAD_DELETED,
    gst_nvevent_parse_pad_deleted,
    gst_nvevent_parse_stream_eos,
    GST_NVEVENT_STREAM_EOS,
    nvds_frame_meta_iterator,
    nvds_obj_meta_iterator,
    nvds_clf_meta_iterator,
    nvds_label_info_iterator,
    nvds_tensor_output_iterator,
    nvds_infer_tensor_meta_to_outputs,
    nvds_add_obj_meta_to_frame,
    nvds_attr_meta_iterator,
    nvds_add_attr_meta_to_obj,
    nvds_remove_obj_attrs,
    nvds_set_selection_type,
    nvds_set_obj_uid,
)
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.meta.type import ObjectSelectionType
from savant.utils.model_registry import ModelObjectRegistry
from savant.utils.source_info import SourceInfoRegistry, SourceInfo
from savant.utils.platform import is_aarch64
from savant.config.schema import PipelineElement, ModelElement, DrawBin
from savant.base.model import ObjectModel, AttributeModel, ComplexModel
from savant.utils.sink_factories import SinkEndOfStream, SinkVideoFrame
from savant.deepstream.element_factory import NvDsElementFactory


class NvDsPipeline(GstPipeline):
    """Base class for managing the DeepStream Pipeline.

    :param name: Pipeline name
    :param source: Pipeline source element
    :param elements: Pipeline elements
    :key frame_width: Processing frame width (after nvstreammux)
    :key frame_height: Processing frame height (after nvstreammux)
    :key batch_size: Primary batch size (nvstreammux batch-size)
    :key output_frame: Whether to include frame in module output, not just metadata
    """

    # pipeline element factory
    _element_factory = NvDsElementFactory()

    def __init__(
        self,
        name: str,
        source: PipelineElement,
        elements: List[PipelineElement],
        **kwargs,
    ):
        # pipeline internal processing frame size
        self._frame_width = kwargs['frame_width']
        self._frame_height = kwargs['frame_height']

        self._batch_size = kwargs['batch_size']
        # Timeout in microseconds
        self._batched_push_timeout = kwargs.get('batched_push_timeout', 2000)

        self._max_parallel_streams = kwargs.get('max_parallel_streams', 64)

        # model artifacts path
        self._model_path = Path(kwargs['model_path'])

        # model-object association storage
        self._model_object_registry = ModelObjectRegistry()

        self._source_adding_lock = Lock()
        self._sources = SourceInfoRegistry()

        # c++ preprocessing class
        self._objects_preprocessing = ObjectsPreprocessing()

        self._internal_attrs = set()

        output_frame = kwargs.get('output_frame')
        if output_frame:
            self._output_frame = True
            self._output_frame_codec = CODEC_BY_NAME[output_frame['codec']]
            self._output_frame_encoder_params = output_frame.get('encoder_params', {})
        else:
            self._output_frame = False
            self._output_frame_codec = None
            self._output_frame_encoder_params = None

        self._demuxer_src_pads: List[Gst.Pad] = []
        self._free_pad_indices: List[int] = []

        if source.element == 'zeromq_source_bin':
            output_alpha_channel = self._output_frame_codec in [
                Codec.PNG,
                Codec.RAW_RGBA,
            ]
            source.properties['convert-jpeg-to-rgb'] = output_alpha_channel
            source.properties['max-parallel-streams'] = self._max_parallel_streams

        super().__init__(name=name, source=source, elements=elements, **kwargs)

    def _add_element(
        self,
        element: PipelineElement,
        with_probes: bool = False,
        link: bool = True,
    ) -> Gst.Element:
        if isinstance(element, ModelElement):
            if element.model.input.preprocess_object_tensor:
                self._objects_preprocessing.add_preprocessing_function(
                    element.name,
                    element.model.input.preprocess_object_tensor.custom_function,
                )
            if isinstance(element.model, (AttributeModel, ComplexModel)):
                for attr in element.model.output.attributes:
                    if attr.internal:
                        self._internal_attrs.add((element.name, attr.name))
        return super()._add_element(element=element, with_probes=with_probes, link=link)

    def _add_source(self, source: PipelineElement):
        source.name = 'source'
        _source = self._add_element(source)
        _source.connect('pad-added', self.on_source_added)

        # Need to suppress EOS on nvstreammux sink pad
        # to prevent pipeline from shutting down
        self._suppress_eos = source.element == 'zeromq_source_bin'
        # nvstreammux is required for NvDs pipeline
        # add queue and set live-source for rtsp
        live_source = source.element == 'uridecodebin' and source.properties[
            'uri'
        ].startswith('rtsp://')
        if live_source:
            self._add_element(PipelineElement('queue'))
        frame_processing_parameters = {
            'width': self._frame_width,
            'height': self._frame_height,
            'batch-size': self._batch_size,
            # Allowed range for batch-size: 1 - 1024
            # Allowed range for buffer-pool-size: 4 - 1024
            'buffer-pool-size': max(4, self._batch_size),
            'batched-push-timeout': self._batched_push_timeout,
            'live-source': live_source,  # True for RTSP or USB camera
            # TODO: remove when the bug with odd will be fixed
            # https://forums.developer.nvidia.com/t/nvstreammux-error-releasing-cuda-memory/219895/3
            'interpolation-method': 6,
        }
        _muxer = self._add_element(
            PipelineElement(
                element='nvstreammux',
                name='muxer',
                properties=frame_processing_parameters,
            ),
            link=False,
        )
        self._logger.info(
            'Pipeline frame processing parameters: %s.', frame_processing_parameters
        )
        # input processor (post-muxer)
        _muxer.get_static_pad('src').add_probe(
            Gst.PadProbeType.BUFFER, self._input_probe, 0
        )
        if self._suppress_eos:
            _muxer.get_static_pad('src').add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                on_pad_event,
                {GST_NVEVENT_PAD_DELETED: self.on_muxer_sink_pad_deleted},
            )

    def on_muxer_sink_pad_deleted(self, pad: Gst.Pad, event: Gst.Event):
        """Send GST_NVEVENT_STREAM_EOS event before GST_NVEVENT_PAD_DELETED.

        GST_NVEVENT_STREAM_EOS is needed to flush data on downstream
        elements.
        """

        pad_idx = gst_nvevent_parse_pad_deleted(event)
        self._logger.debug(
            'Got GST_NVEVENT_PAD_DELETED with source-id %s on %s.%s',
            pad_idx,
            pad.get_parent().get_name(),
            pad.get_name(),
        )
        nv_eos_event = gst_nvevent_new_stream_eos(pad_idx)
        pad.push_event(nv_eos_event)
        return Gst.PadProbeReturn.PASS

    def on_source_added(  # pylint: disable=unused-argument
        self, element: Gst.Element, new_pad: Gst.Pad
    ):
        """Handle adding new video source.

        :param element: The source element that the pad was added to.
        :param new_pad: The pad that has been added.
        """

        # filter out non-video pads
        # new_pad caps can be None, e.g. for zeromq_source_bin
        caps = new_pad.get_current_caps()
        if caps and not caps.get_structure(0).get_name().startswith('video'):
            return

        # new_pad.name example `src_camera1` => source_id == `camera1` (real source_id)
        source_id = pad_to_source_id(new_pad)
        self._logger.debug(
            'Adding source %s. Pad name: %s.', source_id, new_pad.get_name()
        )

        try:
            source_info = self._sources.get_source(source_id)
        except KeyError:
            source_info = self._sources.init_source(source_id)
        else:
            while not source_info.lock.wait(5):
                self._logger.debug(
                    'Waiting source %s to release', source_info.source_id
                )
            source_info.lock.clear()

        self._logger.debug('Ready to add source %s', source_info.source_id)

        # Link elements to source pad only when caps are set
        new_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.CAPS: self._on_source_caps},
            source_info,
        )

    def _on_source_caps(
        self, new_pad: Gst.Pad, event: Gst.Event, source_info: SourceInfo
    ):
        """Handle adding caps to video source pad."""

        new_pad_caps: Gst.Caps = event.parse_caps()
        self._logger.debug(
            'Pad %s.%s has caps %s',
            new_pad.get_parent().get_name(),
            new_pad.get_name(),
            new_pad_caps,
        )

        while source_info.pad_idx is None:
            try:
                with self._source_adding_lock:
                    source_info.pad_idx = self._free_pad_indices.pop(0)
            except IndexError:
                # avro_video_decode_bin already sent EOS for some stream and adding a
                # new one, but the former stream did not complete in this pipeline yet.
                self._logger.warning(
                    'Reached maximum number of streams: %s. '
                    'Waiting resources for source %s.',
                    self._max_parallel_streams,
                    source_info.source_id,
                )
                time.sleep(5)

        with self._source_adding_lock:
            self._sources.update_source(source_info)

            if not source_info.after_demuxer:
                self._add_source_output(source_info)
            input_src_pad = self._add_input_converter(
                new_pad,
                new_pad_caps,
                source_info,
            )
            add_convert_savant_frame_meta_pad_probe(
                input_src_pad,
                True,
            )
            muxer_sink_pad = self._get_muxer_sink_pad(source_info)
            assert input_src_pad.link(muxer_sink_pad) == Gst.PadLinkReturn.OK
            self._pipeline.set_state(Gst.State.PLAYING)

        self._logger.info('Added source %s', source_info.source_id)

        # Video source has been added, removing probe.
        return Gst.PadProbeReturn.REMOVE

    def _add_input_converter(
        self,
        new_pad: Gst.Pad,
        new_pad_caps: Gst.Caps,
        source_info: SourceInfo,
    ) -> Gst.Pad:
        nv_video_converter: Gst.Element = Gst.ElementFactory.make('nvvideoconvert')
        if not is_aarch64():
            nv_video_converter.set_property(
                'nvbuf-memory-type', int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            )
        self._pipeline.add(nv_video_converter)
        nv_video_converter.sync_state_with_parent()
        video_converter_sink: Gst.Pad = nv_video_converter.get_static_pad('sink')
        if not video_converter_sink.query_accept_caps(new_pad_caps):
            self._logger.debug(
                '"nvvideoconvert" cannot accept caps %s.'
                'Inserting "videoconvert" before it.',
                new_pad_caps,
            )
            video_converter: Gst.Element = Gst.ElementFactory.make('videoconvert')
            self._pipeline.add(video_converter)
            video_converter.sync_state_with_parent()
            assert video_converter.link(nv_video_converter)
            video_converter_sink = video_converter.get_static_pad('sink')
            source_info.before_muxer.append(video_converter)

        source_info.before_muxer.append(nv_video_converter)
        # TODO: send EOS to video_converter on unlink if source didn't
        assert new_pad.link(video_converter_sink) == Gst.PadLinkReturn.OK

        return nv_video_converter.get_static_pad('src')

    def _add_source_output(self, source_info: SourceInfo):
        fakesink = super()._add_sink(
            PipelineElement(
                element='fakesink',
                name=f'sink_{source_info.source_id}',
                properties={
                    'sync': 0,
                    'qos': 0,
                    'enable-last-sample': 0,
                },
            ),
            link=False,
            probe_data=source_info,
        )
        fakesink.sync_state_with_parent()

        demuxer_src_pad = self._demuxer_src_pads[source_info.pad_idx]
        fakesink_pad: Gst.Pad = fakesink.get_static_pad('sink')
        fakesink_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.EOS: self.on_last_pad_eos},
            source_info,
        )

        output_queue = self._add_element(PipelineElement('queue'), link=False)
        output_queue.sync_state_with_parent()
        source_info.after_demuxer.append(output_queue)
        assert (
            demuxer_src_pad.link(output_queue.get_static_pad('sink'))
            == Gst.PadLinkReturn.OK
        )

        # demuxer -> queue -> fakesink
        if not self._output_frame:
            assert output_queue.link(fakesink)

        # demuxer -> queue -> nvvideoconvert -> ... -> fakesink
        else:
            output_converter = self._add_element(
                PipelineElement(
                    'nvvideoconvert',
                    properties=(
                        {}
                        if is_aarch64()
                        else {'nvbuf-memory-type': int(pyds.NVBUF_MEM_CUDA_UNIFIED)}
                    ),
                ),
            )
            source_info.after_demuxer.append(output_converter)
            output_converter.sync_state_with_parent()

            if self._output_frame_codec == Codec.RAW_RGBA:
                caps_filter = self._add_element(PipelineElement('capsfilter'))
                caps_filter.set_property(
                    'caps',
                    Gst.Caps.from_string('video/x-raw(memory:NVMM), format=RGBA'),
                )
                source_info.after_demuxer.append(caps_filter)
                caps_filter.sync_state_with_parent()
                assert caps_filter.link(fakesink)

            else:
                add_convert_savant_frame_meta_pad_probe(
                    output_queue.get_static_pad('src'),
                    False,
                )

                encoder = self._add_element(
                    PipelineElement(
                        self._output_frame_codec.value.encoder,
                        properties=self._output_frame_encoder_params,
                    )
                )
                source_info.after_demuxer.append(encoder)
                encoder.sync_state_with_parent()

                # A parser for codecs h264, h265 is added to include
                # the Sequence Parameter Set (SPS) and the Picture Parameter Set (PPS)
                # to IDR frames in the video stream. SPS and PPS are needed
                # to correct recording or playback not from the beginning
                # of the video stream.
                if self._output_frame_codec.value.parser in ['h264parse', 'h265parse']:
                    parser = self._add_element(
                        PipelineElement(
                            self._output_frame_codec.value.parser,
                            properties=({'config-interval': -1}),
                        )
                    )
                    source_info.after_demuxer.append(parser)
                    parser.sync_state_with_parent()
                    assert parser.link(fakesink)
                else:
                    assert encoder.link(fakesink)

        source_info.after_demuxer.append(fakesink)

    def _get_muxer_sink_pad(self, source_info: SourceInfo):
        muxer: Gst.Element = self._pipeline.get_by_name('muxer')
        # sink_N == NvDsFrameMeta.pad_index
        sink_pad_name = f'sink_{source_info.pad_idx}'
        sink_pad: Gst.Pad = muxer.get_static_pad(sink_pad_name)
        if sink_pad is None:
            self._logger.debug(
                'Requesting new sink pad on %s: %s', muxer.get_name(), sink_pad_name
            )
            sink_pad: Gst.Pad = muxer.get_request_pad(sink_pad_name)
            sink_pad.add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                on_pad_event,
                {Gst.EventType.EOS: self.on_muxer_sink_pad_eos},
                source_info.source_id,
            )

        return sink_pad

    def on_muxer_sink_pad_eos(self, pad: Gst.Pad, event: Gst.Event, source_id: str):
        """Processes EOS event on muxer sink pad."""
        self._logger.debug(
            'Got EOS on pad %s.%s', pad.get_parent().get_name(), pad.get_name()
        )
        source_info = self._sources.get_source(source_id)
        GLib.idle_add(self._remove_input_elements, source_info, pad)
        return (
            Gst.PadProbeReturn.DROP if self._suppress_eos else Gst.PadProbeReturn.PASS
        )

    def _remove_input_elements(
        self,
        source_info: SourceInfo,
        muxer_sink_pad: Gst.Pad,
    ):
        self._logger.debug(
            'Removing input elements for source %s', source_info.source_id
        )
        for elem in source_info.before_muxer:
            self._logger.debug('Removing element %s', elem.get_name())
            elem.set_locked_state(True)
            elem.set_state(Gst.State.NULL)
            self._pipeline.remove(elem)
        source_info.before_muxer = []
        muxer: Gst.Element = muxer_sink_pad.get_parent()
        self._logger.debug(
            'Releasing pad %s.%s; source: %s',
            muxer_sink_pad.get_parent().get_name(),
            muxer_sink_pad.get_name(),
            source_info.source_id,
        )
        # Releasing muxer.sink pad to trigger nv-pad-deleted event on muxer.src pad
        muxer.release_request_pad(muxer_sink_pad)
        self._logger.debug(
            'Input elements for source %s removed', source_info.source_id
        )
        return False

    def on_demuxer_src_pad_eos(self, pad: Gst.Pad, event: Gst.Event):
        """Processes EOS events on demuxer src pad."""
        pad_idx = gst_nvevent_parse_stream_eos(event)
        if pad_idx is None or pad != self._demuxer_src_pads[pad_idx]:
            # nvstreamdemux redirects GST_NVEVENT_STREAM_EOS on each src pad
            return Gst.PadProbeReturn.PASS
        self._logger.debug(
            'Got GST_NVEVENT_STREAM_EOS on %s.%s',
            pad.get_parent().get_name(),
            pad.get_name(),
        )
        peer: Gst.Pad = pad.get_peer()
        if peer is not None:
            self._logger.debug(
                'Unlinking %s.%s from %s.%s',
                peer.get_parent().get_name(),
                peer.get_name(),
                pad.get_parent().get_name(),
                pad.get_name(),
            )
            pad.unlink(peer)
            self._logger.debug(
                'Sending EOS to %s.%s',
                peer.get_parent().get_name(),
                peer.get_name(),
            )
            peer.send_event(Gst.Event.new_eos())
        return Gst.PadProbeReturn.DROP

    def on_last_pad_eos(self, pad: Gst.Pad, event: Gst.Event, source_info: SourceInfo):
        """Process EOS on last pad."""
        self._logger.debug(
            'Got EOS on pad %s.%s', pad.get_parent().get_name(), pad.get_name()
        )
        GLib.idle_add(self._remove_output_elements, source_info)

        self._queue.put(SinkEndOfStream(source_info.source_id))

        return (
            Gst.PadProbeReturn.DROP if self._suppress_eos else Gst.PadProbeReturn.PASS
        )

    def _remove_output_elements(self, source_info: SourceInfo):
        """Process EOS on last pad."""
        self._logger.debug(
            'Removing output elements for source %s', source_info.source_id
        )
        for elem in source_info.after_demuxer:
            self._logger.debug('Removing element %s', elem.get_name())
            elem.set_locked_state(True)
            elem.set_state(Gst.State.NULL)
            self._pipeline.remove(elem)
        source_info.after_demuxer = []
        self._logger.debug(
            'Output elements for source %s removed', source_info.source_id
        )

        self._sources.remove_source(source_info)

        self._free_pad_indices.append(source_info.pad_idx)
        source_info.pad_idx = None
        self._logger.debug('Releasing lock for source %s', source_info.source_id)
        source_info.lock.set()
        self._logger.info(
            'Resources for source %s has been released.', source_info.source_id
        )
        return False

    def _add_sink(
        self,
        sink: Optional[PipelineElement] = None,
        link: bool = True,
        probe_data: Any = None,
    ) -> Gst.Element:
        """Adds sink elements."""

        # add drawbin if frame should be in module output and there is no drawbin
        if self._output_frame and not [
            e for e, _ in self.elements if isinstance(e, DrawBin)
        ]:
            self._add_element(DrawBin())

        demuxer = self._add_element(
            PipelineElement(
                element='nvstreamdemux',
                name='demuxer',
            ),
            link=link,
        )
        self._demuxer_src_pads = self.allocate_demuxer_pads(
            demuxer, self._max_parallel_streams
        )
        self._free_pad_indices = list(range(len(self._demuxer_src_pads)))
        demuxer.get_static_pad('sink').add_probe(
            Gst.PadProbeType.BUFFER, self.update_frame_meta
        )

    def allocate_demuxer_pads(self, demuxer: Gst.Element, n_pads: int):
        """Allocate a fixed number of demuxer src pads."""
        pads = []
        for pad_idx in range(n_pads):
            pad: Gst.Pad = demuxer.get_request_pad(f'src_{pad_idx}')
            pad.add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                on_pad_event,
                {GST_NVEVENT_STREAM_EOS: self.on_demuxer_src_pad_eos},
            )
            pads.append(pad)
        return pads

    def prepare_input(self, buffer: Gst.Buffer):
        """Input meta processor.

        :param buffer: gstreamer buffer that is being processed.
        """

        self._logger.debug('Preparing input for buffer with PTS %s.', buffer.pts)
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            source_id = self._sources.get_id_by_pad_index(nvds_frame_meta.pad_index)
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            frame_idx = savant_frame_meta.idx if savant_frame_meta else None
            frame_pts = nvds_frame_meta.buf_pts

            self._logger.debug(
                'Preparing input for frame %s of source %s.',
                nvds_frame_meta.buf_pts,
                source_id,
            )
            frame_meta = metadata_get_frame_meta(source_id, frame_idx, frame_pts)

            # add external objects to nvds meta
            for obj_meta in frame_meta.metadata['objects']:
                obj_key = self._model_object_registry.model_object_key(
                    obj_meta['model_name'], obj_meta['label']
                )
                # skip frame, will be added with proper width/height
                if obj_key == 'frame':
                    continue
                # obj_key was only registered if
                # it was required by the pipeline model elements (this case)
                # or equaled the output object of one of the pipeline model elements
                if self._model_object_registry.is_model_object_key_registered(obj_key):
                    (
                        model_uid,
                        class_id,
                    ) = self._model_object_registry.get_model_object_ids(obj_key)
                    if obj_meta['bbox']['angle']:
                        scaled_bbox = scale_rbbox(
                            bboxes=np.array(
                                [
                                    [
                                        obj_meta['bbox']['xc'],
                                        obj_meta['bbox']['yc'],
                                        obj_meta['bbox']['width'],
                                        obj_meta['bbox']['height'],
                                        obj_meta['bbox']['angle'],
                                    ]
                                ]
                            ),
                            scale_factor_x=self._frame_width,
                            scale_factor_y=self._frame_height,
                        )[0]
                        selection_type = ObjectSelectionType.ROTATED_BBOX
                    else:
                        scaled_bbox = (
                            obj_meta['bbox']['xc'] * self._frame_width,
                            obj_meta['bbox']['yc'] * self._frame_height,
                            obj_meta['bbox']['width'] * self._frame_width,
                            obj_meta['bbox']['height'] * self._frame_height,
                            obj_meta['bbox']['angle'],
                        )
                        selection_type = ObjectSelectionType.REGULAR_BBOX

                    nvds_add_obj_meta_to_frame(
                        batch_meta=nvds_batch_meta,
                        frame_meta=nvds_frame_meta,
                        selection_type=selection_type,
                        class_id=class_id,
                        gie_uid=model_uid,
                        bbox=scaled_bbox,
                        object_id=obj_meta['object_id'],
                        obj_label=obj_key,
                        confidence=obj_meta['confidence'],
                    )

            # add primary frame object
            obj_label = 'frame'
            model_uid, class_id = self._model_object_registry.get_model_object_ids(
                obj_label
            )
            nvds_add_obj_meta_to_frame(
                batch_meta=nvds_batch_meta,
                frame_meta=nvds_frame_meta,
                selection_type=ObjectSelectionType.REGULAR_BBOX,
                class_id=class_id,
                gie_uid=model_uid,
                # tuple(xc, yc, width, height, angle)
                bbox=(
                    self._frame_width / 2,
                    self._frame_height / 2,
                    self._frame_width,
                    self._frame_height,
                    0,
                ),
                obj_label=obj_label,
                # confidence should be bigger than tracker minDetectorConfidence
                # to prevent the tracker from deleting the object
                # use tracker display-tracking-id=0 to avoid labelling
                confidence=0.999,
            )

            nvds_frame_meta.bInferDone = True  # required for tracker (DS 6.0)

    def prepare_output(
        self,
        buffer: Gst.Buffer,
        source_info: SourceInfo,
    ):
        """Enqueue output messages based on frame meta.

        :param buffer: gstreamer buffer that is being processed.
        :param source_info: output source info
        """

        self._logger.debug(
            'Preparing output for buffer with PTS %s for source %s.',
            buffer.pts,
            source_info.source_id,
        )
        for idx, pts, frame, keyframe in (
            self._iterate_frames_from_gst_buffer(buffer)
            if self._output_frame and self._output_frame_codec != Codec.RAW_RGBA
            else self._iterate_frames_from_nvds_batch(buffer)
        ):
            self._logger.debug(
                'Preparing output for frame %s with PTS %s of source %s.',
                idx,
                pts,
                source_info.source_id,
            )
            frame_meta = metadata_pop_frame_meta(source_info.source_id, idx, pts)
            self._queue.put(
                SinkVideoFrame(
                    source_info.source_id,
                    frame_meta,
                    self._frame_width,
                    self._frame_height,
                    frame,
                    (
                        self._output_frame_codec.value
                        if self._output_frame_codec is not None
                        else None
                    ),
                    keyframe=keyframe,
                )
            )

    def _iterate_frames_from_gst_buffer(self, buffer: Gst.Buffer):
        """Iterate frames from Gst.Buffer.

        :return: generator of (IDX, PTS, frame, is_keyframe)
        """

        # get frame if required for output
        frame = buffer.extract_dup(0, buffer.get_size()) if self._output_frame else None
        savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
        frame_idx = savant_frame_meta.idx if savant_frame_meta else None
        frame_pts = buffer.pts
        is_keyframe = not buffer.has_flags(Gst.BufferFlags.DELTA_UNIT)
        yield frame_idx, frame_pts, frame, is_keyframe

    def _iterate_frames_from_nvds_batch(self, buffer: Gst.Buffer):
        """Iterate frames from NvDs batch.

        NvDs batch contains raw RGBA frames. They are all keyframes.

        :return: generator of (IDX, PTS, frame, is_keyframe)
        """

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            # get frame if required for output
            frame = (
                pyds.get_nvds_buf_surface(
                    hash(buffer), nvds_frame_meta.batch_id
                ).tobytes()
                if self._output_frame
                else None
            )
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            frame_idx = savant_frame_meta.idx if savant_frame_meta else None
            frame_pts = nvds_frame_meta.buf_pts
            # Any frame is keyframe since it was not encoded
            yield frame_idx, frame_pts, frame, True

    def update_frame_meta(self, pad: Gst.Pad, info: Gst.PadProbeInfo):
        """Prepare frame meta for output."""
        buffer: Gst.Buffer = info.get_buffer()
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))

        # convert output meta
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            # use consecutive numbers for object_id in case there is no tracker
            object_ids = defaultdict(int)
            # first iteration to correct object_id
            for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                # correct object_id (track_id)
                if nvds_obj_meta.object_id == UNTRACKED_OBJECT_ID:
                    nvds_obj_meta.object_id = object_ids[nvds_obj_meta.obj_label]
                    object_ids[nvds_obj_meta.obj_label] += 1

            # will extend source metadata
            source_id = self._sources.get_id_by_pad_index(nvds_frame_meta.pad_index)
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            frame_idx = savant_frame_meta.idx if savant_frame_meta else None
            frame_pts = nvds_frame_meta.buf_pts
            frame_meta = metadata_get_frame_meta(source_id, frame_idx, frame_pts)

            # second iteration to collect module objects
            for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                # skip fake primary frame object
                if nvds_obj_meta.obj_label == 'frame':
                    continue

                obj_meta = nvds_obj_meta_output_converter(
                    nvds_obj_meta, self._frame_width, self._frame_height
                )
                for attr_meta_list in nvds_attr_meta_iterator(
                    frame_meta=nvds_frame_meta, obj_meta=nvds_obj_meta
                ):
                    for attr_meta in attr_meta_list:
                        if (
                            attr_meta.element_name,
                            attr_meta.name,
                        ) not in self._internal_attrs:
                            obj_meta['attributes'].append(
                                nvds_attr_meta_output_converter(attr_meta)
                            )
                nvds_remove_obj_attrs(nvds_frame_meta, nvds_obj_meta)
                frame_meta.metadata['objects'].append(obj_meta)

            metadata_add_frame_meta(source_id, frame_idx, frame_pts, frame_meta)

        return Gst.PadProbeReturn.PASS

    def _is_model_input_object(
        self, element: ModelElement, nvds_obj_meta: pyds.NvDsObjectMeta
    ):
        model_uid, class_id = self._model_object_registry.get_model_object_ids(
            element.model.input.object
        )
        return (
            nvds_obj_meta.unique_component_id == model_uid
            and nvds_obj_meta.class_id == class_id
        )

    def prepare_element_input(self, element: PipelineElement, buffer: Gst.Buffer):
        """Model input preprocessing.

        :param element: element that this probe was added to.
        :param buffer: gstreamer buffer that is being processed.
        """
        if not isinstance(element, ModelElement):
            return

        model = element.model
        if (
            not model.input.preprocess_object_meta
            and not model.input.preprocess_object_tensor
        ):
            return

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if model.input.preprocess_object_meta:
            for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
                for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                    if not self._is_model_input_object(element, nvds_obj_meta):
                        continue
                    # TODO: Unify and also switch to the box representation system
                    #  through the center point during meta preprocessing.
                    bbox = pyds.NvBbox_Coords()  # why not?
                    bbox.left = nvds_obj_meta.rect_params.left
                    bbox.top = nvds_obj_meta.rect_params.top
                    bbox.width = nvds_obj_meta.rect_params.width
                    bbox.height = nvds_obj_meta.rect_params.height

                    parent_bbox = pyds.NvBbox_Coords()
                    if nvds_obj_meta.parent:
                        parent_bbox.left = nvds_obj_meta.parent.rect_params.left
                        parent_bbox.top = nvds_obj_meta.parent.rect_params.top
                        parent_bbox.width = nvds_obj_meta.parent.rect_params.width
                        parent_bbox.height = nvds_obj_meta.parent.rect_params.height
                    else:
                        parent_bbox.left = 0
                        parent_bbox.top = 0
                        parent_bbox.width = self._frame_width
                        parent_bbox.height = self._frame_height

                    bbox = model.input.preprocess_object_meta(
                        bbox, parent_bbox=parent_bbox
                    )

                    rect_params = nvds_obj_meta.rect_params
                    rect_params.left = bbox.left
                    rect_params.top = bbox.top
                    rect_params.width = bbox.width
                    rect_params.height = bbox.height

        elif model.input.preprocess_object_tensor:
            model_uid, class_id = self._model_object_registry.get_model_object_ids(
                model.input.object
            )
            self._objects_preprocessing.preprocessing(
                element.name,
                hash(buffer),
                model_uid,
                class_id,
                model.input.preprocess_object_tensor.padding[0],
                model.input.preprocess_object_tensor.padding[1],
            )

    def prepare_element_output(self, element: PipelineElement, buffer: Gst.Buffer):
        """Model output postprocessing.

        :param element: element that this probe was added to.
        :param buffer: gstreamer buffer that is being processed.
        """
        if not isinstance(element, ModelElement):
            return

        model_uid = self._model_object_registry.get_model_uid(element.name)
        model: Union[
            NvInferRotatedObjectDetector,
            NvInferDetector,
            NvInferAttributeModel,
        ] = element.model
        is_complex_model = isinstance(model, ComplexModel)
        is_object_model = isinstance(model, ObjectModel)

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
                # convert custom model output and save meta
                if model.output.converter:
                    if not self._is_model_input_object(element, nvds_obj_meta):
                        continue
                    parent_nvds_obj_meta = nvds_obj_meta
                    for tensor_meta in nvds_tensor_output_iterator(
                        parent_nvds_obj_meta, gie_uid=model_uid
                    ):
                        # parse and post-process model output
                        output_layers = nvds_infer_tensor_meta_to_outputs(
                            tensor_meta=tensor_meta,
                            layer_names=model.output.layer_names,
                        )
                        outputs = model.output.converter(
                            *output_layers,
                            model=model,
                            roi=(
                                parent_nvds_obj_meta.rect_params.left,
                                parent_nvds_obj_meta.rect_params.top,
                                parent_nvds_obj_meta.rect_params.width,
                                parent_nvds_obj_meta.rect_params.height,
                            ),
                        )
                        # for object/complex models output - `bbox_tensor` and
                        # `selected_bboxes` - indices of selected bboxes and meta
                        # for attribute/complex models output - `values`
                        bbox_tensor, selected_bboxes, values = None, None, None
                        # complex model
                        if is_complex_model:
                            # output converter returns tensor and attribute values
                            bbox_tensor, values = outputs
                            assert bbox_tensor.shape[0] == len(
                                values
                            ), 'Number of detected boxes and attributes do not match.'

                        # object model
                        elif is_object_model:
                            # output converter returns tensor with
                            # (class_id, confidence, xc, yc, width, height, [angle]),
                            # coordinates in roi scale (parent object scale)
                            bbox_tensor = outputs

                        # attribute model
                        else:
                            # output converter returns attribute values
                            values = outputs

                        if bbox_tensor is not None and bbox_tensor.shape[0] > 0:

                            if bbox_tensor.shape[1] == 6:  # no angle
                                # xc -> left, yc -> top
                                bbox_tensor[:, 2] -= bbox_tensor[:, 4] / 2
                                bbox_tensor[:, 3] -= bbox_tensor[:, 5] / 2

                                # clip
                                # width to right, height to bottom
                                bbox_tensor[:, 4] += bbox_tensor[:, 2]
                                bbox_tensor[:, 5] += bbox_tensor[:, 3]
                                # clip
                                bbox_tensor[:, 2][bbox_tensor[:, 2] < 0.0] = 0.0
                                bbox_tensor[:, 3][bbox_tensor[:, 3] < 0.0] = 0.0
                                bbox_tensor[:, 4][
                                    bbox_tensor[:, 4] > self._frame_width - 1.0
                                ] = (self._frame_width - 1.0)
                                bbox_tensor[:, 5][
                                    bbox_tensor[:, 5] > self._frame_height - 1.0
                                ] = (self._frame_height - 1.0)

                                # right to width, bottom to height
                                bbox_tensor[:, 4] -= bbox_tensor[:, 2]
                                bbox_tensor[:, 5] -= bbox_tensor[:, 3]

                                # left -> xc , top-> yc
                                bbox_tensor[:, 2] += bbox_tensor[:, 4] / 2
                                bbox_tensor[:, 3] += bbox_tensor[:, 5] / 2

                                # add 0 angle
                                bbox_tensor = np.concatenate(
                                    [bbox_tensor, np.zeros((bbox_tensor.shape[0], 1))],
                                    axis=1,
                                )

                            # add index column to further filter attribute values
                            bbox_tensor = np.concatenate(
                                [
                                    bbox_tensor,
                                    np.arange(bbox_tensor.shape[0]).reshape(-1, 1),
                                ],
                                axis=1,
                            )

                            selected_bboxes = []
                            for obj in model.output.objects:
                                cls_bbox_tensor = bbox_tensor[
                                    bbox_tensor[:, 0] == obj.class_id
                                ]
                                if cls_bbox_tensor.shape[0] == 0:
                                    continue
                                if obj.selector:
                                    cls_bbox_tensor = obj.selector(cls_bbox_tensor)

                                obj_label = (
                                    self._model_object_registry.model_object_key(
                                        element.name, obj.label
                                    )
                                )
                                for bbox in cls_bbox_tensor:
                                    # add NvDsObjectMeta
                                    _nvds_obj_meta = nvds_add_obj_meta_to_frame(
                                        batch_meta=nvds_batch_meta,
                                        frame_meta=nvds_frame_meta,
                                        selection_type=model.output.selection_type,
                                        class_id=obj.class_id,
                                        gie_uid=model_uid,
                                        bbox=bbox[2:7],
                                        parent=parent_nvds_obj_meta,
                                        obj_label=obj_label,
                                        confidence=bbox[1],
                                    )
                                    selected_bboxes.append(
                                        (int(bbox[7]), _nvds_obj_meta)
                                    )

                        if values:
                            if is_complex_model:
                                values = [
                                    v
                                    for i, v in enumerate(values)
                                    if i in {i for i, o in selected_bboxes}
                                ]
                            else:
                                selected_bboxes = [(0, nvds_obj_meta)]
                                values = [values]
                            for (_, _nvds_obj_meta), _values in zip(
                                selected_bboxes, values
                            ):
                                for attr_name, value, confidence in _values:
                                    nvds_add_attr_meta_to_obj(
                                        frame_meta=nvds_frame_meta,
                                        obj_meta=_nvds_obj_meta,
                                        element_name=element.name,
                                        name=attr_name,
                                        value=value,
                                        confidence=confidence,
                                    )

                # regular object model (detector)
                # correct nvds_obj_meta.obj_label
                elif is_object_model:
                    if nvds_obj_meta.unique_component_id == model_uid:
                        for obj in model.output.objects:
                            if nvds_obj_meta.class_id == obj.class_id:
                                nvds_set_selection_type(
                                    obj_meta=nvds_obj_meta,
                                    selection_type=ObjectSelectionType.REGULAR_BBOX,
                                )
                                nvds_set_obj_uid(
                                    frame_meta=nvds_frame_meta, obj_meta=nvds_obj_meta
                                )
                                nvds_obj_meta.obj_label = (
                                    self._model_object_registry.model_object_key(
                                        element.name, obj.label
                                    )
                                )
                                break

                # regular attribute model (classifier)
                # convert nvds_clf_meta to attr_meta
                else:
                    for nvds_clf_meta in nvds_clf_meta_iterator(nvds_obj_meta):
                        if nvds_clf_meta.unique_component_id != model_uid:
                            continue
                        for attr, label_info in zip(
                            model.output.attributes,
                            nvds_label_info_iterator(nvds_clf_meta),
                        ):
                            nvds_add_attr_meta_to_obj(
                                frame_meta=nvds_frame_meta,
                                obj_meta=nvds_obj_meta,
                                element_name=element.name,
                                name=attr.name,
                                value=label_info.result_label,
                                confidence=label_info.result_prob,
                            )

                # restore nvds_obj_meta.rect_params if there was preprocessing
                if (
                    model.input.preprocess_object_meta
                    or model.input.preprocess_object_tensor
                ) and self._is_model_input_object(element, nvds_obj_meta):
                    bbox_coords = nvds_obj_meta.detector_bbox_info.org_bbox_coords
                    if nvds_obj_meta.tracker_bbox_info.org_bbox_coords.width > 0:
                        bbox_coords = nvds_obj_meta.tracker_bbox_info.org_bbox_coords
                    rect_params = nvds_obj_meta.rect_params
                    rect_params.left = bbox_coords.left
                    rect_params.top = bbox_coords.top
                    rect_params.width = bbox_coords.width
                    rect_params.height = bbox_coords.height

        # restore frame
        if model.input.preprocess_object_tensor:
            self._objects_preprocessing.restore_frame(hash(buffer))
