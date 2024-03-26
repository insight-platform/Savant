"""DeepStream pipeline."""
import logging
import time
from collections import defaultdict
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import pyds
from pygstsavantframemeta import (
    add_convert_savant_frame_meta_pad_probe,
    add_pad_probe_to_move_batch,
    add_pad_probe_to_move_frame,
    add_pad_probe_to_pack_and_move_frames,
    add_pad_probe_to_unpack_and_move_batch,
    gst_buffer_get_savant_batch_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)
from savant_rs.pipeline2 import VideoPipeline, VideoPipelineConfiguration
from savant_rs.primitives import EndOfStream, VideoFrame
from savant_rs.primitives.geometry import RBBox

from savant.base.input_preproc import ObjectsPreprocessing
from savant.base.model import AttributeModel, ComplexModel
from savant.config.schema import (
    BufferQueuesParameters,
    DrawFunc,
    FrameParameters,
    ModelElement,
    Pipeline,
    PipelineElement,
    PyFuncElement,
    TelemetryParameters,
)
from savant.deepstream.buffer_processor import (
    NvDsBufferProcessor,
    create_buffer_processor,
)
from savant.deepstream.element_factory import NvDsElementFactory
from savant.deepstream.metadata import (
    nvds_attr_meta_output_converter,
    nvds_obj_bbox_output_converter,
    nvds_obj_meta_output_converter,
)
from savant.deepstream.nvinfer.processor import NvInferProcessor
from savant.deepstream.source_output import create_source_output
from savant.deepstream.utils.attribute import (
    nvds_attr_meta_iterator,
    nvds_remove_obj_attrs,
)
from savant.deepstream.utils.event import (
    GST_NVEVENT_STREAM_EOS,
    gst_nvevent_parse_stream_eos,
)
from savant.deepstream.utils.iterator import (
    nvds_frame_meta_iterator,
    nvds_obj_meta_iterator,
)
from savant.deepstream.utils.object import nvds_is_empty_object_meta
from savant.deepstream.utils.pipeline import (
    add_queues_to_pipeline,
    build_metrics_exporter,
    build_pipeline_stages,
    get_pipeline_element_stages,
    init_tracing,
)
from savant.gstreamer import GLib, Gst  # noqa:F401
from savant.gstreamer.pipeline import GstPipeline
from savant.gstreamer.utils import add_buffer_probe, on_pad_event, pad_to_source_id
from savant.meta.constants import PRIMARY_OBJECT_KEY, UNTRACKED_OBJECT_ID
from savant.utils.platform import is_aarch64
from savant.utils.sink_factories import SinkEndOfStream
from savant.utils.source_info import Resolution, SourceInfo, SourceInfoRegistry


class NvDsPipeline(GstPipeline):
    """Base class for managing the DeepStream Pipeline.

    :param name: Pipeline name.
    :param pipeline_cfg: Pipeline config.
    :key frame: Processing frame parameters (after nvstreammux).
    :key batch_size: Primary batch size (nvstreammux batch-size).
    :key output_frame: Whether to include frame in module output, not just metadata.
    """

    _element_factory = NvDsElementFactory()

    def __init__(
        self,
        name: str,
        pipeline_cfg: Pipeline,
        **kwargs,
    ):
        # pipeline internal processing frame params
        self._frame_params: FrameParameters = kwargs['frame']

        self._batch_size = kwargs['batch_size']
        self._stream_buffer_pool_size = kwargs.get('stream_buffer_pool_size')
        self._muxer_buffer_pool_size = kwargs.get('muxer_buffer_pool_size')

        # Timeout in microseconds
        self._batched_push_timeout = kwargs.get('batched_push_timeout', 2000)

        self._max_parallel_streams = kwargs.get('max_parallel_streams', 64)

        self._egress_queue_length = kwargs.get('egress_queue_length')
        self._egress_queue_byte_size = kwargs.get('egress_queue_byte_size')
        self._egress_queue_properties = {
            'max-size-buffers': self._egress_queue_length,
            'max-size-bytes': self._egress_queue_byte_size,
            'max-size-time': 0,
        }

        self._source_adding_lock = Lock()
        self._sources = SourceInfoRegistry()

        # c++ preprocessing class
        self._objects_preprocessing = ObjectsPreprocessing(self._batch_size)

        self._internal_attrs = set()
        telemetry: TelemetryParameters = kwargs['telemetry']
        init_tracing(name, telemetry.tracing)

        output_frame = kwargs.get('output_frame')
        self._pass_through_mode = bool(output_frame) and output_frame['codec'] == 'copy'
        draw_func: Optional[DrawFunc] = kwargs.get('draw_func')
        if draw_func is not None and output_frame:
            pipeline_cfg.elements.append(draw_func)

        self._demuxer_src_pads: List[Gst.Pad] = []
        self._free_pad_indices: List[int] = []
        self._last_nvevent_stream_eos_seqnum: Dict[int, int] = {}
        self._muxer: Optional[Gst.Element] = None

        if pipeline_cfg.source.element == 'zeromq_source_bin':
            pipeline_cfg.source.properties.update(
                {
                    'max-parallel-streams': self._max_parallel_streams,
                    'pipeline-source-stage-name': 'source',
                    'pipeline-demux-stage-name': 'source-demuxer',
                    'pipeline-decoder-stage-name': 'decode',
                }
            )
            shutdown_auth = kwargs.get('shutdown_auth')
            if shutdown_auth is not None:
                pipeline_cfg.source.properties['shutdown-auth'] = shutdown_auth

        buffer_queues: Optional[BufferQueuesParameters] = kwargs.get('buffer_queues')
        if buffer_queues is not None:
            add_queues_to_pipeline(pipeline_cfg, buffer_queues)

        self._element_stages = get_pipeline_element_stages(pipeline_cfg)
        pipeline_stages = build_pipeline_stages(self._element_stages)
        if telemetry.tracing.root_span_name is not None:
            root_span_name = telemetry.tracing.root_span_name
        else:
            root_span_name = name

        self._video_pipeline = VideoPipeline(
            root_span_name,
            pipeline_stages,
            build_video_pipeline_conf(telemetry),
        )
        self._video_pipeline.sampling_period = telemetry.tracing.sampling_period
        self._metrics_exporter = build_metrics_exporter(
            self._video_pipeline,
            telemetry.metrics,
        )

        self._source_output = create_source_output(
            frame_params=self._frame_params,
            output_frame=output_frame,
            video_pipeline=self._video_pipeline,
            queue_properties=self._egress_queue_properties,
        )

        # nvjpegdec decoder is selected in decodebin according to the rank, but
        # the plugin doesn't support some jpg
        #  https://forums.developer.nvidia.com/t/nvvideoconvert-memory-compatibility-error/226138;
        # Set the rank to NONE for the plugin to not use it.
        if is_aarch64():
            factory = Gst.ElementFactory.find('nvjpegdec')
            factory.set_rank(Gst.Rank.NONE)

        super().__init__(name, pipeline_cfg, **kwargs)

    def _build_buffer_processor(self, queue: Queue) -> NvDsBufferProcessor:
        """Create buffer processor."""

        return create_buffer_processor(
            queue=queue,
            sources=self._sources,
            frame_params=self._frame_params,
            source_output=self._source_output,
            video_pipeline=self._video_pipeline,
            pass_through_mode=self._pass_through_mode,
        )

    def add_element(
        self,
        element: PipelineElement,
        link: bool = True,
        element_idx: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> Gst.Element:
        gst_element = super().add_element(
            element=element,
            link=link,
            element_idx=element_idx,
        )

        if isinstance(element, ModelElement):
            if isinstance(element.model, (AttributeModel, ComplexModel)):
                for attr in element.model.output.attributes:
                    if attr.internal:
                        self._internal_attrs.add((element.name, attr.name))
            nvinfer = NvInferProcessor(
                element,
                self._objects_preprocessing,
                self._frame_params,
                self._video_pipeline,
            )
            if nvinfer.preproc is not None:
                add_buffer_probe(gst_element.get_static_pad('sink'), nvinfer.preproc)
            if nvinfer.postproc is not None:
                add_buffer_probe(gst_element.get_static_pad('src'), nvinfer.postproc)

        if element_idx is not None:
            if isinstance(element, PyFuncElement):
                gst_element.set_property('pipeline', self._video_pipeline)
                gst_element.set_property('stream-pool-size', self._batch_size)
                if self._metrics_exporter is not None:
                    gst_element.set_property('metrics-exporter', self._metrics_exporter)
            # TODO: add stage names to element config?
            if isinstance(element_idx, int):
                stage = self._element_stages[element_idx]
            else:
                stage = self._element_stages[element_idx[0]][element_idx[1]]
            add_pad_probe_to_move_batch(
                gst_element.get_static_pad('sink'),
                self._video_pipeline,
                stage,
            )

        return gst_element

    def on_startup(self):
        if self._metrics_exporter is not None:
            self._metrics_exporter.start()
        super().on_startup()

    def before_shutdown(self):
        super().before_shutdown()
        self._disable_eos_suppression()

    def on_shutdown(self):
        self._video_pipeline.log_final_fps()
        if self._metrics_exporter is not None:
            self._metrics_exporter.stop()
        super().on_shutdown()

    def _on_shutdown_signal(self, element: Gst.Element):
        """Handle shutdown signal."""

        self._logger.info('Received shutdown signal from %s.', element.get_name())
        self._disable_eos_suppression()
        Thread(target=self._handle_shutdown_signal, daemon=True).start()

    def _handle_shutdown_signal(self):
        while True:
            with self._source_adding_lock:
                if not self._sources.has_sources():
                    break
            self._logger.debug('Waiting for sources to release')
            time.sleep(0.1)
        if not self._is_running:
            self._logger.info('Pipeline was shut down already.')
            return

        with self._source_adding_lock:
            self._logger.info('Shutting down the pipeline.')
            # We need to add a fakesink element to the pipeline and receive EOS
            # with it to shut down the pipeine.
            self._logger.debug('Adding fakesink to the pipeline')
            fakesink = self.add_element(
                PipelineElement(
                    'fakesink',
                    properties={
                        'sync': 0,
                        'qos': 0,
                        'enable-last-sample': 0,
                    },
                ),
                link=False,
            )
            fakesink_pad: Gst.Pad = fakesink.get_static_pad('sink')
            fakesink.sync_state_with_parent()
            fakesink_pad.send_event(Gst.Event.new_eos())

    def _disable_eos_suppression(self):
        self._logger.debug(
            'Turning off "drop-pipeline-eos" of %s', self._muxer.get_name()
        )
        self._suppress_eos = False
        self._muxer.set_property('drop-pipeline-eos', False)

    # Source
    def _add_source(self, source: PipelineElement):
        source.name = 'source'
        _source = self.add_element(source)
        if source.element == 'zeromq_source_bin':
            _source.set_property('pipeline', self._video_pipeline)
            _source.set_property('pass-through-mode', self._pass_through_mode)
            _source.connect('shutdown', self._on_shutdown_signal)
            add_frames_to_pipeline = False
        else:
            add_frames_to_pipeline = True
        _source.connect(
            'pad-added',
            self.on_source_added,
            add_frames_to_pipeline,
        )

        # Need to suppress EOS on nvstreammux sink pad
        # to prevent pipeline from shutting down
        self._suppress_eos = source.element == 'zeromq_source_bin'
        # nvstreammux is required for NvDs pipeline
        # add queue and set live-source for rtsp
        live_source = source.element == 'uridecodebin' and source.properties[
            'uri'
        ].startswith('rtsp://')
        if live_source:
            self.add_element(PipelineElement('queue'))
        self._create_muxer(live_source)

    # Sink
    def _add_sink(
        self,
        sink: Optional[PipelineElement] = None,
        link: bool = True,
        probe_data: Any = None,
    ) -> Gst.Element:
        """Adds sink elements."""

        self._create_demuxer(link)
        self._free_pad_indices = list(range(len(self._demuxer_src_pads)))

    # Input
    def on_source_added(  # pylint: disable=unused-argument
        self,
        element: Gst.Element,
        new_pad: Gst.Pad,
        add_frames_to_pipeline: bool,
    ):
        """Handle adding new video source.

        :param element: The source element that the pad was added to.
        :param new_pad: The pad that has been added.
        :param add_frames_to_pipeline: Whether to add frames to pipeline i.e.
            add an element savant_rs_add_frames after the source element.
        """

        if not self._is_running:
            self._logger.info(
                'Pipeline is not running. Do not add source for pad %s.',
                new_pad.get_name(),
            )
            return

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
            while self._is_running and not source_info.lock.wait(5):
                self._logger.debug(
                    'Waiting source %s to release', source_info.source_id
                )
            source_info.lock.clear()

        if not self._is_running:
            self._logger.info(
                'Pipeline is not running. Cancel adding source %s.',
                source_id,
            )
            return

        self._logger.debug('Ready to add source %s', source_info.source_id)

        # Link elements to source pad only when caps are set
        new_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.CAPS: self._on_source_caps},
            source_info,
            add_frames_to_pipeline,
        )

    def _on_source_caps(
        self,
        new_pad: Gst.Pad,
        event: Gst.Event,
        source_info: SourceInfo,
        add_frames_to_pipeline: bool,
    ):
        """Handle adding caps to video source pad."""

        try:
            new_pad_caps: Gst.Caps = event.parse_caps()
            self._logger.debug(
                'Pad %s.%s has caps %s',
                new_pad.get_parent().get_name(),
                new_pad.get_name(),
                new_pad_caps,
            )
            caps_struct: Gst.Structure = new_pad_caps.get_structure(0)
            parsed, width = caps_struct.get_int('width')
            assert parsed, f'Failed to parse "width" property of caps "{new_pad_caps}"'
            parsed, height = caps_struct.get_int('height')
            assert parsed, f'Failed to parse "height" property of caps "{new_pad_caps}"'

            while source_info.pad_idx is None:
                self._check_pipeline_is_running()
                try:
                    with self._source_adding_lock:
                        source_info.pad_idx = self._free_pad_indices.pop(0)
                except IndexError:
                    # savant_rs_video_decode_bin already sent EOS for some stream and adding a
                    # new one, but the former stream did not complete in this pipeline yet.
                    self._logger.warning(
                        'Reached maximum number of streams: %s. '
                        'Waiting resources for source %s.',
                        self._max_parallel_streams,
                        source_info.source_id,
                    )
                    time.sleep(5)

            with self._source_adding_lock:
                self._check_pipeline_is_running()
                source_info.src_resolution = Resolution(width, height)
                source_info.add_scale_transformation = (
                    self._frame_params.width != width
                    or self._frame_params.height != height
                )
                self._sources.update_source(source_info)

                if not source_info.after_demuxer:
                    self._add_source_output(source_info)
                input_src_pad = self._add_input_converter(
                    new_pad,
                    new_pad_caps,
                    source_info,
                    add_frames_to_pipeline,
                )
                self._check_pipeline_is_running()
                add_pad_probe_to_move_frame(
                    input_src_pad,
                    self._video_pipeline,
                    'muxer',
                )
                add_convert_savant_frame_meta_pad_probe(
                    input_src_pad,
                    True,
                )
                self._link_to_muxer(input_src_pad, source_info)
                self._check_pipeline_is_running()
                self._pipeline.set_state(Gst.State.PLAYING)

        except PipelineIsNotRunningError:
            self._logger.info(
                'Pipeline is not running. Cancel adding source %s.',
                source_info.source_id,
            )
            return Gst.PadProbeReturn.REMOVE

        self._logger.info('Added source %s', source_info.source_id)

        # Video source has been added, removing probe.
        return Gst.PadProbeReturn.REMOVE

    def _add_input_converter(
        self,
        new_pad: Gst.Pad,
        new_pad_caps: Gst.Caps,
        source_info: SourceInfo,
        add_frames_to_pipeline: bool,
    ) -> Gst.Pad:
        self._check_pipeline_is_running()
        if add_frames_to_pipeline:
            # Add savant frames to VideoPipeline when source element is not zeromq_source_bin
            # (e.g. uridecodebin).
            # Cannot add frames with a probe since Gst.Buffer is not writable,
            # and it's impossible to make it writable in a probe.
            savant_rs_add_frames = self._element_factory.create(
                PipelineElement(
                    'savant_rs_add_frames',
                    properties={
                        'source-id': source_info.source_id,
                        'pipeline-stage-name': 'source',
                    },
                )
            )
            savant_rs_add_frames.set_property('pipeline', self._video_pipeline)
            self._pipeline.add(savant_rs_add_frames)
            source_info.before_muxer.append(savant_rs_add_frames)
            savant_rs_add_frames.sync_state_with_parent()
            savant_rs_add_frames_sink: Gst.Pad = savant_rs_add_frames.get_static_pad(
                'sink'
            )
            assert new_pad.link(savant_rs_add_frames_sink) == Gst.PadLinkReturn.OK
            new_pad = savant_rs_add_frames.get_static_pad('src')

        add_pad_probe_to_move_frame(new_pad, self._video_pipeline, 'source-convert')

        nv_video_converter_props = {}
        if self._stream_buffer_pool_size is not None:
            nv_video_converter_props['output-buffers'] = self._stream_buffer_pool_size
        if is_aarch64() and new_pad_caps.get_structure(0).get_value('format') == 'RGB':
            #   https://forums.developer.nvidia.com/t/buffer-transform-failed-for-nvvideoconvert-for-num-input-channels-num-output-channels-on-jetson/237578
            #   https://forums.developer.nvidia.com/t/nvvideoconvert-buffer-transform-failed-on-jetson/261370
            self._logger.info(
                'Input stream is RGB, using compute-hw=1 as recommended by Nvidia'
            )
            nv_video_converter_props['compute-hw'] = 1
        if self._frame_params.padding:
            dest_crop = ':'.join(
                str(x)
                for x in [
                    self._frame_params.padding.left,
                    self._frame_params.padding.top,
                    self._frame_params.width,
                    self._frame_params.height,
                ]
            )
            nv_video_converter_props['dest-crop'] = dest_crop

        nv_video_converter = self._element_factory.create(
            PipelineElement('nvvideoconvert', properties=nv_video_converter_props)
        )

        self._check_pipeline_is_running()
        self._pipeline.add(nv_video_converter)
        nv_video_converter.sync_state_with_parent()
        video_converter_sink: Gst.Pad = nv_video_converter.get_static_pad('sink')
        if not video_converter_sink.query_accept_caps(new_pad_caps):
            self._logger.debug(
                '"nvvideoconvert" cannot accept caps %s. '
                'Inserting "videoconvert" before it.',
                new_pad_caps,
            )
            self._check_pipeline_is_running()
            video_converter = self._element_factory.create(
                PipelineElement('videoconvert')
            )
            self._pipeline.add(video_converter)
            video_converter.sync_state_with_parent()
            assert video_converter.link(nv_video_converter)
            video_converter_sink = video_converter.get_static_pad('sink')
            source_info.before_muxer.append(video_converter)

        source_info.before_muxer.append(nv_video_converter)
        self._check_pipeline_is_running()
        # TODO: send EOS to video_converter on unlink if source didn't
        assert new_pad.link(video_converter_sink) == Gst.PadLinkReturn.OK

        self._check_pipeline_is_running()
        capsfilter = self._element_factory.create(
            PipelineElement(
                'capsfilter',
                properties={
                    'caps': (
                        'video/x-raw(memory:NVMM), format=RGBA, '
                        f'width={self._frame_params.total_width}, '
                        f'height={self._frame_params.total_height}'
                    ),
                },
            )
        )
        add_pad_probe_to_move_frame(
            capsfilter.get_static_pad('sink'),
            self._video_pipeline,
            'source-capsfilter',
        )

        capsfilter.set_state(Gst.State.PLAYING)
        self._pipeline.add(capsfilter)
        source_info.before_muxer.append(capsfilter)
        assert nv_video_converter.link(capsfilter)

        return capsfilter.get_static_pad('src')

    def _remove_input_elements(self, source_info: SourceInfo):
        self._logger.debug(
            'Removing input elements for source %s', source_info.source_id
        )

        try:
            for elem in source_info.before_muxer:
                self._check_pipeline_is_running()
                self._logger.debug('Removing element %s', elem.get_name())
                elem.set_locked_state(True)
                elem.set_state(Gst.State.NULL)
                self._pipeline.remove(elem)
            source_info.before_muxer = []

        except PipelineIsNotRunningError:
            self._logger.info(
                'Pipeline is not running. Cancel removing input elements for source %s.',
                source_info.source_id,
            )
            return False

        self._logger.debug(
            'Input elements for source %s removed', source_info.source_id
        )
        return False

    # Output
    def _add_source_output(self, source_info: SourceInfo):
        self._check_pipeline_is_running()
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
        fakesink_pad: Gst.Pad = fakesink.get_static_pad('sink')
        fakesink.sync_state_with_parent()

        self._check_pipeline_is_running()
        fakesink_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.EOS: self.on_last_pad_eos},
            source_info,
        )

        self._check_pipeline_is_running()
        output_queue = self.add_element(
            PipelineElement('queue', properties=self._egress_queue_properties),
            link=False,
        )
        output_queue.sync_state_with_parent()
        source_info.after_demuxer.append(output_queue)
        output_queue_sink_pad: Gst.Pad = output_queue.get_static_pad('sink')
        add_pad_probe_to_move_frame(
            output_queue_sink_pad,
            self._video_pipeline,
            'output-queue',
        )
        self._link_demuxer_src_pad(output_queue_sink_pad, source_info)

        self._check_pipeline_is_running()
        output_pad: Gst.Pad = self._source_output.add_output(
            pipeline=self,
            source_info=source_info,
            input_pad=output_queue.get_static_pad('src'),
        )
        self._check_pipeline_is_running()
        assert output_pad.link(fakesink_pad) == Gst.PadLinkReturn.OK

        source_info.after_demuxer.append(fakesink)

    def _remove_output_elements(self, source_info: SourceInfo):
        """Process EOS on last pad."""
        self._logger.debug(
            'Removing output elements for source %s', source_info.source_id
        )

        try:
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

        except PipelineIsNotRunningError:
            self._logger.info(
                'Pipeline is not running. Cancel removing output elements for source %s.',
                source_info.source_id,
            )
            return False

        finally:
            source_info.pad_idx = None
            self._logger.debug('Releasing lock for source %s', source_info.source_id)
            source_info.lock.set()

        self._logger.info(
            'Resources for source %s has been released.', source_info.source_id
        )
        return False

    def on_last_pad_eos(self, pad: Gst.Pad, event: Gst.Event, source_info: SourceInfo):
        """Process EOS on last pad."""
        self._logger.debug(
            'Got EOS on pad %s.%s', pad.get_parent().get_name(), pad.get_name()
        )
        self._buffer_processor.on_eos(source_info)

        try:
            self._check_pipeline_is_running()
            GLib.idle_add(self._remove_output_elements, source_info)
        except PipelineIsNotRunningError:
            self._logger.info(
                'Pipeline is not running. Do not remove output elements for source %s.',
                source_info.source_id,
            )

        self._queue.put(SinkEndOfStream(EndOfStream(source_info.source_id)))

        return (
            Gst.PadProbeReturn.DROP if self._suppress_eos else Gst.PadProbeReturn.PASS
        )

    def update_frame_meta(self, pad: Gst.Pad, info: Gst.PadProbeInfo):
        """Prepare frame meta for output."""
        buffer: Gst.Buffer = info.get_buffer()

        self._logger.debug('Prepare meta output for buffer with PTS %s', buffer.pts)

        savant_batch_meta = gst_buffer_get_savant_batch_meta(buffer)
        if savant_batch_meta is None:
            self._logger.warning(
                'Failed to update frame meta for batch at buffer %s. '
                'Batch has no Savant Frame Meta.',
                buffer.pts,
            )
            return Gst.PadProbeReturn.PASS

        batch_id = savant_batch_meta.idx
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
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            if savant_frame_meta is None:
                self._logger.warning(
                    'Failed to update frame meta for frame %s at buffer %s. '
                    'Frame has no Savant Frame Meta.',
                    nvds_frame_meta.buf_pts,
                    buffer.pts,
                )
                continue

            frame_idx = savant_frame_meta.idx
            video_frame: VideoFrame
            video_frame, video_frame_span = self._video_pipeline.get_batched_frame(
                batch_id,
                frame_idx,
            )

            with video_frame_span.nested_span('update-frame-meta'):
                self._update_meta_for_single_frame(
                    frame_idx=frame_idx,
                    nvds_frame_meta=nvds_frame_meta,
                    video_frame=video_frame,
                )

        return Gst.PadProbeReturn.PASS

    def _update_meta_for_single_frame(
        self,
        frame_idx: int,
        nvds_frame_meta: pyds.NvDsFrameMeta,
        video_frame: VideoFrame,
    ):
        frame_pts = nvds_frame_meta.buf_pts
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(
                'Preparing output for frame of source %s with IDX %s and PTS %s.',
                video_frame.source_id,
                frame_idx,
                frame_pts,
            )

        # second iteration to collect module objects
        nvds_object_id_map = {}  # nvds_obj_meta.object_id -> video_object.id
        parents = {}  # video_object.id -> nvds_obj_meta.parent.object_id
        for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
            # collect obj attributes
            attributes = []
            for attr_meta in nvds_attr_meta_iterator(
                frame_meta=nvds_frame_meta, obj_meta=nvds_obj_meta
            ):
                if (
                    attr_meta.element_name,
                    attr_meta.name,
                ) not in self._internal_attrs:
                    attributes.append(nvds_attr_meta_output_converter(attr_meta))
            nvds_remove_obj_attrs(nvds_frame_meta, nvds_obj_meta)

            # check if primary object
            if nvds_obj_meta.obj_label == PRIMARY_OBJECT_KEY:
                bbox = nvds_obj_bbox_output_converter(nvds_obj_meta, self._frame_params)
                dest_res_bbox = RBBox(
                    self._frame_params.output_width / 2,
                    self._frame_params.output_height / 2,
                    self._frame_params.output_width,
                    self._frame_params.output_height,
                )
                if not bbox.almost_eq(dest_res_bbox, 1e-6):
                    if self._logger.isEnabledFor(logging.DEBUG):
                        self._logger.debug(
                            'Adding primary object, bbox %s != dest res bbox %s.',
                            bbox,
                            dest_res_bbox,
                        )
                elif attributes:
                    self._logger.debug('Adding primary object, attributes not empty.')
                else:
                    # skip primary object equal to frame
                    # and with no attached attributes
                    self._logger.debug('Skipping empty primary object.')
                    continue

            # convert nvds object meta to savant-rs object meta
            obj_meta = nvds_obj_meta_output_converter(
                nvds_obj_meta, self._frame_params, video_frame
            )
            nvds_object_id_map[nvds_obj_meta.object_id] = obj_meta.id
            if (
                not nvds_is_empty_object_meta(nvds_obj_meta.parent)
                and nvds_obj_meta.parent.obj_label != PRIMARY_OBJECT_KEY
            ):
                parents[obj_meta.id] = nvds_obj_meta.parent.object_id
            # add obj attributes
            for attr in attributes:
                obj_meta.set_attribute(attr)
            if self._logger.isEnabledFor(logging.TRACE):
                self._logger.trace(
                    'Collecting object (frame src %s, IDX %s, PTS %s): %s',
                    video_frame.source_id,
                    frame_idx,
                    frame_pts,
                    obj_meta,
                )

        self._logger.debug('Setting parents to objects.')
        for obj_id, parent_id in parents.items():
            video_frame.set_parent_by_id(obj_id, nvds_object_id_map[parent_id])

    # Muxer
    def _create_muxer(self, live_source: bool) -> Gst.Element:
        """Create nvstreammux element and add it into pipeline.

        :param live_source: Whether source is live or not.
        """

        frame_processing_params = {
            'width': self._frame_params.total_width,
            'height': self._frame_params.total_height,
            # Allowed range for batch-size: 1 - 1024
            'batch-size': self._batch_size,
            # Allowed range for buffer-pool-size: 4 - 1024
            'batched-push-timeout': self._batched_push_timeout,
            'live-source': live_source,  # True for RTSP or USB camera
            # TODO: remove when the bug with odd will be fixed
            # https://forums.developer.nvidia.com/t/nvstreammux-error-releasing-cuda-memory/219895/3
            'interpolation-method': 6,
            'drop-pipeline-eos': self._suppress_eos,
        }

        if self._muxer_buffer_pool_size is not None:
            frame_processing_params['buffer-pool-size'] = max(
                4, self._muxer_buffer_pool_size
            )

        if not is_aarch64():
            frame_processing_params['nvbuf-memory-type'] = int(
                pyds.NVBUF_MEM_CUDA_UNIFIED
            )

        self._muxer = self.add_element(
            PipelineElement(
                element='nvstreammux',
                name='muxer',
                properties=frame_processing_params,
            ),
            link=False,
        )
        self._logger.info(
            'Pipeline frame processing parameters: %s.', frame_processing_params
        )
        # input processor (post-muxer)
        muxer_src_pad: Gst.Pad = self._muxer.get_static_pad('src')
        add_pad_probe_to_pack_and_move_frames(
            muxer_src_pad,
            self._video_pipeline,
            'prepare-input',
        )
        add_buffer_probe(muxer_src_pad, self._buffer_processor.prepare_input)

        return self._muxer

    def _link_to_muxer(self, pad: Gst.Pad, source_info: SourceInfo):
        """Link src pad to muxer.

        :param pad: Src pad to connect.
        :param source_info: Video source info.
        """

        muxer_sink_pad = self._request_muxer_sink_pad(source_info)
        self._check_pipeline_is_running()
        pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.EOS: self._on_muxer_sink_pad_peer_eos},
            source_info.source_id,
        )
        assert pad.link(muxer_sink_pad) == Gst.PadLinkReturn.OK

    def _request_muxer_sink_pad(self, source_info: SourceInfo) -> Gst.Pad:
        """Request sink pad from muxer.

        :param source_info: Video source info.
        """

        self._check_pipeline_is_running()
        # sink_N == NvDsFrameMeta.pad_index
        sink_pad_name = f'sink_{source_info.pad_idx}'
        sink_pad: Gst.Pad = self._muxer.get_static_pad(sink_pad_name)
        if sink_pad is None:
            self._logger.debug(
                'Requesting new sink pad on %s: %s',
                self._muxer.get_name(),
                sink_pad_name,
            )
            self._check_pipeline_is_running()
            sink_pad: Gst.Pad = self._muxer.get_request_pad(sink_pad_name)

        return sink_pad

    def _on_muxer_sink_pad_peer_eos(
        self, pad: Gst.Pad, event: Gst.Event, source_id: str
    ):
        """Processes EOS event on a peer of muxer sink pad."""

        self._logger.debug(
            'Got EOS on pad %s.%s', pad.get_parent().get_name(), pad.get_name()
        )
        source_info = self._sources.get_source(source_id)
        try:
            self._check_pipeline_is_running()
            GLib.idle_add(self._remove_input_elements, source_info)
        except PipelineIsNotRunningError:
            self._logger.info(
                'Pipeline is not running. Do not remove output elements for source %s.',
                source_info.source_id,
            )
        return Gst.PadProbeReturn.PASS

    # Demuxer
    def _create_demuxer(self, link: bool) -> Gst.Element:
        """Create nvstreamdemux element and add it into pipeline.

        :param link: Whether to automatically link demuxer to the last pipeline element.
        """

        demuxer = self.add_element(
            PipelineElement(
                element='nvstreamdemux',
                name='demuxer',
            ),
            link=link,
        )
        self._demuxer_src_pads = self._allocate_demuxer_pads(
            demuxer, self._max_parallel_streams
        )
        sink_peer_pad: Gst.Pad = demuxer.get_static_pad('sink').get_peer()
        add_pad_probe_to_move_batch(
            sink_peer_pad,
            self._video_pipeline,
            'update-frame-meta',
        )
        sink_peer_pad.add_probe(Gst.PadProbeType.BUFFER, self.update_frame_meta)
        add_pad_probe_to_unpack_and_move_batch(
            sink_peer_pad,
            self._video_pipeline,
            'demuxer',
        )
        return demuxer

    def _allocate_demuxer_pads(self, demuxer: Gst.Element, n_pads: int):
        """Allocate a fixed number of demuxer src pads."""

        pads = []
        for pad_idx in range(n_pads):
            pad: Gst.Pad = demuxer.get_request_pad(f'src_{pad_idx}')
            add_convert_savant_frame_meta_pad_probe(pad, False)
            pad.add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                on_pad_event,
                {GST_NVEVENT_STREAM_EOS: self._on_demuxer_src_pad_eos},
            )
            pads.append(pad)
        return pads

    def _on_demuxer_src_pad_eos(self, pad: Gst.Pad, event: Gst.Event):
        """Processes EOS events on demuxer src pad."""

        pad_idx = gst_nvevent_parse_stream_eos(event)
        if pad != self._demuxer_src_pads[pad_idx]:
            # nvstreamdemux redirects GST_NVEVENT_STREAM_EOS on each src pad
            return Gst.PadProbeReturn.DROP

        if event.get_seqnum() <= self._last_nvevent_stream_eos_seqnum.get(pad_idx, -1):
            # This event has already been processed
            return Gst.PadProbeReturn.DROP
        self._last_nvevent_stream_eos_seqnum[pad_idx] = event.get_seqnum()

        self._logger.debug(
            'Got GST_NVEVENT_STREAM_EOS on %s.%s',
            pad.get_parent().get_name(),
            pad.get_name(),
        )

        try:
            self._check_pipeline_is_running()
            peer: Gst.Pad = pad.get_peer()
            if peer is not None:
                self._logger.debug(
                    'Unlinking %s.%s from %s.%s',
                    peer.get_parent().get_name(),
                    peer.get_name(),
                    pad.get_parent().get_name(),
                    pad.get_name(),
                )
                self._check_pipeline_is_running()
                pad.unlink(peer)
                self._logger.debug(
                    'Sending EOS to %s.%s',
                    peer.get_parent().get_name(),
                    peer.get_name(),
                )
                self._check_pipeline_is_running()
                peer.send_event(Gst.Event.new_eos())

        except PipelineIsNotRunningError:
            self._logger.info(
                'Pipeline is not running. Cancel unlinking demuxer pad %s.',
                pad.get_name(),
            )

        return Gst.PadProbeReturn.DROP

    def _link_demuxer_src_pad(self, pad: Gst.Pad, source_info: SourceInfo):
        """Link demuxer src pad to some sink pad.

        :param pad: Connect demuxer src pad to this sink pad.
        :param source_info: Video source info.
        """

        self._check_pipeline_is_running()
        demuxer_src_pad = self._demuxer_src_pads[source_info.pad_idx]
        assert demuxer_src_pad.link(pad) == Gst.PadLinkReturn.OK

    def _check_pipeline_is_running(self):
        """Raise an exception if pipeline is not running."""

        if not self._is_running:
            raise PipelineIsNotRunningError('Pipeline is not running')


def build_video_pipeline_conf(telemetry_params: TelemetryParameters):
    conf = VideoPipelineConfiguration()
    conf.append_frame_meta_to_otlp_span = (
        telemetry_params.tracing.append_frame_meta_to_span
    )
    conf.frame_period = (
        telemetry_params.metrics.frame_period
        if telemetry_params.metrics.frame_period is not None
        else None
    )
    conf.timestamp_period = (
        int(telemetry_params.metrics.time_period * 1000)
        if telemetry_params.metrics.time_period is not None
        else None
    )
    conf.collection_history = telemetry_params.metrics.history

    return conf


class PipelineIsNotRunningError(Exception):
    """Pipeline is not running."""
