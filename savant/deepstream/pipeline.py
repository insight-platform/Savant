"""DeepStream pipeline."""
from collections import defaultdict
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, List, Optional
import time
import logging
import pyds
from savant_rs import init_jaeger_tracer
from savant_rs.pipeline import VideoPipeline, VideoPipelineStagePayloadType
from savant_rs.primitives import EndOfStream, IdCollisionResolutionPolicy, VideoFrame
from savant_rs.primitives.geometry import RBBox

from savant.base.input_preproc import ObjectsPreprocessing

from pygstsavantframemeta import (
    add_convert_savant_frame_meta_pad_probe,
    gst_buffer_get_savant_frame_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)

from savant.deepstream.buffer_processor import (
    NvDsBufferProcessor,
    create_buffer_processor,
)
from savant.deepstream.ds_probes import (
    add_move_batch_as_is_pad_probe,
    add_move_batch_to_frames_pad_probe,
    add_move_frames_to_batch_pad_probe,
)
from savant.deepstream.gst_probes import add_move_frame_as_is_pad_probe
from savant.deepstream.source_output import (
    SourceOutputWithFrame,
    create_source_output,
)
from savant.deepstream.utils.pipeline import add_queues_to_pipeline
from savant.gstreamer import Gst, GLib  # noqa:F401
from savant.gstreamer.pipeline import GstPipeline
from savant.deepstream.metadata import (
    nvds_obj_meta_output_converter,
    nvds_attr_meta_output_converter,
)
from savant.gstreamer.utils import on_pad_event, pad_to_source_id
from savant.deepstream.utils import (
    gst_nvevent_parse_stream_eos,
    GST_NVEVENT_STREAM_EOS,
    nvds_frame_meta_iterator,
    nvds_obj_meta_iterator,
    nvds_attr_meta_iterator,
    nvds_remove_obj_attrs,
)
from savant.meta.constants import UNTRACKED_OBJECT_ID, PRIMARY_OBJECT_KEY
from savant.utils.fps_meter import FPSMeter
from savant.utils.source_info import SourceInfoRegistry, SourceInfo, Resolution
from savant.utils.platform import is_aarch64
from savant.config.schema import (
    BufferQueuesParameters,
    Pipeline,
    PipelineElement,
    ModelElement,
    FrameParameters,
    DrawFunc,
    PyFuncElement,
    TelemetryParameters,
)
from savant.base.model import AttributeModel, ComplexModel
from savant.utils.sink_factories import SinkEndOfStream


class NvDsPipeline(GstPipeline):
    """Base class for managing the DeepStream Pipeline.

    :param name: Pipeline name.
    :param pipeline_cfg: Pipeline config.
    :key frame: Processing frame parameters (after nvstreammux).
    :key batch_size: Primary batch size (nvstreammux batch-size).
    :key output_frame: Whether to include frame in module output, not just metadata.
    """

    def __init__(
        self,
        name: str,
        pipeline_cfg: Pipeline,
        **kwargs,
    ):
        # pipeline internal processing frame size
        self._frame_params: FrameParameters = kwargs['frame']

        self._batch_size = kwargs['batch_size']
        # Timeout in microseconds
        self._batched_push_timeout = kwargs.get('batched_push_timeout', 2000)

        self._max_parallel_streams = kwargs.get('max_parallel_streams', 64)

        # model artifacts path
        self._model_path = Path(kwargs['model_path'])

        self._source_adding_lock = Lock()
        self._sources = SourceInfoRegistry()

        # c++ preprocessing class
        self._objects_preprocessing = ObjectsPreprocessing()

        self._internal_attrs = set()
        telemetry: TelemetryParameters = kwargs['telemetry']
        _init_telemetry(name, telemetry)

        self._video_pipeline = VideoPipeline(name)
        self._video_pipeline.sampling_period = telemetry.sampling_period
        if telemetry.root_span_name is not None:
            self._video_pipeline.root_span_name = telemetry.root_span_name
        else:
            self._video_pipeline.root_span_name = f'{name}-root'

        output_frame = kwargs.get('output_frame')
        self._source_output = create_source_output(
            frame_params=self._frame_params,
            output_frame=output_frame,
            video_pipeline=self._video_pipeline,
        )

        draw_func: Optional[DrawFunc] = kwargs.get('draw_func')
        if draw_func is not None and isinstance(
            self._source_output, SourceOutputWithFrame
        ):
            pipeline_cfg.elements.append(draw_func)

        self._demuxer_src_pads: List[Gst.Pad] = []
        self._free_pad_indices: List[int] = []
        self._muxer: Optional[Gst.Element] = None

        if pipeline_cfg.source.element == 'zeromq_source_bin':
            pipeline_cfg.source.properties.update(
                {
                    'max-parallel-streams': self._max_parallel_streams,
                    'pipeline-source-stage-name': 'source',
                    'pipeline-decoder-stage-name': 'decode',
                }
            )

        buffer_queues: Optional[BufferQueuesParameters] = kwargs.get('buffer_queues')
        if buffer_queues is not None:
            add_queues_to_pipeline(pipeline_cfg, buffer_queues)

        # nvjpegdec decoder is selected in decodebin according to the rank, but
        # there are problems with the plugin:
        # 1) https://forums.developer.nvidia.com/t/nvvideoconvert-memory-compatibility-error/226138;
        # 2) jpeg to png conversion gives incorrect alpha channel;
        # 3) memory type mismatch, even though we use the same
        #  nvbuf-memory-type for nvvideoconvert and nvstreammux downstream
        # Set the rank to NONE for the plugin to not use it.
        factory = Gst.ElementFactory.find('nvjpegdec')
        factory.set_rank(Gst.Rank.NONE)

        super().__init__(name, pipeline_cfg, **kwargs)

    def _build_buffer_processor(
        self,
        queue: Queue,
        fps_meter: FPSMeter,
    ) -> NvDsBufferProcessor:
        """Create buffer processor."""

        return create_buffer_processor(
            queue=queue,
            fps_meter=fps_meter,
            sources=self._sources,
            objects_preprocessing=self._objects_preprocessing,
            frame_params=self._frame_params,
            source_output=self._source_output,
            video_pipeline=self._video_pipeline,
        )

    def add_element(
        self,
        element: PipelineElement,
        with_probes: bool = False,
        link: bool = True,
        pipeline_stage: bool = False,
    ) -> Gst.Element:
        if isinstance(element, ModelElement):
            if element.model.input.preprocess_object_image:
                self._objects_preprocessing.add_preprocessing_function(
                    element_name=element.name,
                    preprocessing_func=element.model.input.preprocess_object_image,
                )
            if isinstance(element.model, (AttributeModel, ComplexModel)):
                for attr in element.model.output.attributes:
                    if attr.internal:
                        self._internal_attrs.add((element.name, attr.name))
        gst_element = super().add_element(
            element=element,
            with_probes=with_probes,
            link=link,
            pipeline_stage=pipeline_stage,
        )
        if pipeline_stage:
            if isinstance(element, PyFuncElement):
                stage = f'pyfunc/{element.module}.{element.class_name}'
                gst_element.set_property('pipeline', self._video_pipeline)
            else:
                stage = gst_element.get_name()
            self._video_pipeline.add_stage(stage, VideoPipelineStagePayloadType.Batch)
            add_move_batch_as_is_pad_probe(
                gst_element.get_static_pad('sink'),
                self._video_pipeline,
                stage,
            )

        return gst_element

    def before_shutdown(self):
        super().before_shutdown()
        self._logger.debug(
            'Turning off "drop-pipeline-eos" of %s', self._muxer.get_name()
        )
        self._muxer.set_property('drop-pipeline-eos', False)

    # Source
    def _add_source(self, source: PipelineElement):
        source.name = 'source'
        _source = self.add_element(source)
        for stage in ['source', 'source-convert', 'source-capsfilter']:
            self._video_pipeline.add_stage(stage, VideoPipelineStagePayloadType.Frame)
        if source.element == 'zeromq_source_bin':
            self._video_pipeline.add_stage(
                'decode',
                VideoPipelineStagePayloadType.Frame,
            )
            _source.set_property('pipeline', self._video_pipeline)
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
                add_move_frame_as_is_pad_probe(
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
            savant_rs_add_frames: Gst.Element = Gst.ElementFactory.make(
                'savant_rs_add_frames'
            )
            savant_rs_add_frames.set_property('source-id', source_info.source_id)
            savant_rs_add_frames.set_property('pipeline', self._video_pipeline)
            savant_rs_add_frames.set_property('pipeline-stage-name', 'source')
            self._pipeline.add(savant_rs_add_frames)
            source_info.before_muxer.append(savant_rs_add_frames)
            savant_rs_add_frames.sync_state_with_parent()
            savant_rs_add_frames_sink: Gst.Pad = savant_rs_add_frames.get_static_pad(
                'sink'
            )
            assert new_pad.link(savant_rs_add_frames_sink) == Gst.PadLinkReturn.OK
            new_pad = savant_rs_add_frames.get_static_pad('src')

        add_move_frame_as_is_pad_probe(new_pad, self._video_pipeline, 'source-convert')

        nv_video_converter: Gst.Element = Gst.ElementFactory.make('nvvideoconvert')
        if not is_aarch64():
            nv_video_converter.set_property(
                'nvbuf-memory-type', int(pyds.NVBUF_MEM_CUDA_UNIFIED)
            )
        elif new_pad_caps.get_structure(0).get_value("format") == "RGB":
            #   https://forums.developer.nvidia.com/t/buffer-transform-failed-for-nvvideoconvert-for-num-input-channels-num-output-channels-on-jetson/237578
            #   https://forums.developer.nvidia.com/t/nvvideoconvert-buffer-transform-failed-on-jetson/261370
            self._logger.info(
                "Input stream is RGB, using  compute-hw=1 as recommended by Nvidia"
            )
            nv_video_converter.set_property('compute-hw', 1)
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
            nv_video_converter.set_property('dest-crop', dest_crop)

        self._check_pipeline_is_running()
        self._pipeline.add(nv_video_converter)
        nv_video_converter.sync_state_with_parent()
        video_converter_sink: Gst.Pad = nv_video_converter.get_static_pad('sink')
        if not video_converter_sink.query_accept_caps(new_pad_caps):
            self._logger.debug(
                '"nvvideoconvert" cannot accept caps %s.'
                'Inserting "videoconvert" before it.',
                new_pad_caps,
            )
            self._check_pipeline_is_running()
            video_converter: Gst.Element = Gst.ElementFactory.make('videoconvert')
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
        capsfilter: Gst.Element = Gst.ElementFactory.make('capsfilter')
        capsfilter.set_property(
            'caps',
            Gst.Caps.from_string(
                'video/x-raw(memory:NVMM), format=RGBA, '
                f'width={self._frame_params.total_width}, '
                f'height={self._frame_params.total_height}'
            ),
        )
        add_move_frame_as_is_pad_probe(
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
        output_queue = self.add_element(PipelineElement('queue'), link=False)
        output_queue.sync_state_with_parent()
        source_info.after_demuxer.append(output_queue)
        output_queue_sink_pad: Gst.Pad = output_queue.get_static_pad('sink')
        add_move_frame_as_is_pad_probe(
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

        savant_batch_meta = gst_buffer_get_savant_frame_meta(buffer)
        if savant_batch_meta is None:
            self._logger.warning(
                'Failed to update frame meta for batch at buffer %s. '
                'Batch has no Savant Frame Meta.',
                buffer.pts,
            )
            return Gst.PadProbeReturn.PASS

        batch_id = savant_batch_meta.idx if savant_batch_meta else None
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

            frame_idx = savant_frame_meta.idx if savant_frame_meta else None
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

        parents = {}

        # second iteration to collect module objects
        for nvds_obj_meta in nvds_obj_meta_iterator(nvds_frame_meta):
            obj_meta, parent_id = nvds_obj_meta_output_converter(
                nvds_frame_meta,
                nvds_obj_meta,
                self._frame_params,
            )
            if parent_id is not None:
                parents[obj_meta.id] = parent_id
            for attr_meta_list in nvds_attr_meta_iterator(
                frame_meta=nvds_frame_meta, obj_meta=nvds_obj_meta
            ):
                for attr_meta in attr_meta_list:
                    if (
                        attr_meta.element_name,
                        attr_meta.name,
                    ) not in self._internal_attrs:
                        obj_meta.set_attribute(
                            nvds_attr_meta_output_converter(attr_meta)
                        )
            nvds_remove_obj_attrs(nvds_frame_meta, nvds_obj_meta)

            # skip empty primary object that equals to frame
            if nvds_obj_meta.obj_label == PRIMARY_OBJECT_KEY:
                bbox = obj_meta.detection_box
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
                elif obj_meta.attributes:
                    self._logger.debug('Adding primary object, attributes not empty.')
                else:
                    self._logger.debug('Skipping empty primary object.')
                    continue
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug(
                    'Collecting object (frame src %s, IDX %s, PTS %s): %s',
                    video_frame.source_id,
                    frame_idx,
                    frame_pts,
                    obj_meta,
                )
            video_frame.add_object(obj_meta, IdCollisionResolutionPolicy.Overwrite)

        self._logger.debug('Setting parents to objects')
        for obj_id, parent_id in parents.items():
            if self._logger.isEnabledFor(logging.DEBUG):
                self._logger.debug('%s is a parent of %s', parent_id, obj_id)
            video_frame.set_parent(obj_id, parent_id)

    # Muxer
    def _create_muxer(self, live_source: bool) -> Gst.Element:
        """Create nvstreammux element and add it into pipeline.

        :param live_source: Whether source is live or not.
        """

        frame_processing_parameters = {
            'width': self._frame_params.total_width,
            'height': self._frame_params.total_height,
            'batch-size': self._batch_size,
            # Allowed range for batch-size: 1 - 1024
            # Allowed range for buffer-pool-size: 4 - 1024
            'buffer-pool-size': max(4, self._batch_size),
            'batched-push-timeout': self._batched_push_timeout,
            'live-source': live_source,  # True for RTSP or USB camera
            # TODO: remove when the bug with odd will be fixed
            # https://forums.developer.nvidia.com/t/nvstreammux-error-releasing-cuda-memory/219895/3
            'interpolation-method': 6,
            'drop-pipeline-eos': self._suppress_eos,
        }
        if not is_aarch64():
            frame_processing_parameters['nvbuf-memory-type'] = int(
                pyds.NVBUF_MEM_CUDA_UNIFIED
            )

        self._muxer = self.add_element(
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
        self._video_pipeline.add_stage('muxer', VideoPipelineStagePayloadType.Frame)
        self._video_pipeline.add_stage(
            'prepare-input',
            VideoPipelineStagePayloadType.Batch,
        )
        muxer_src_pad: Gst.Pad = self._muxer.get_static_pad('src')
        add_move_frames_to_batch_pad_probe(
            muxer_src_pad,
            self._video_pipeline,
            'prepare-input',
        )
        muxer_src_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            self._buffer_processor.input_probe,
        )

        return self._muxer

    def _link_to_muxer(self, pad: Gst.Pad, source_info: SourceInfo):
        """Link src pad to muxer.

        :param pad: Src pad to connect.
        :param source_info: Video source info.
        """

        muxer_sink_pad = self._request_muxer_sink_pad(source_info)
        self._check_pipeline_is_running()
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
            sink_pad.add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                on_pad_event,
                {Gst.EventType.EOS: self._on_muxer_sink_pad_eos},
                source_info.source_id,
            )

        return sink_pad

    def _on_muxer_sink_pad_eos(self, pad: Gst.Pad, event: Gst.Event, source_id: str):
        """Processes EOS event on muxer sink pad."""

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
        self._video_pipeline.add_stage(
            'update-frame-meta', VideoPipelineStagePayloadType.Batch
        )
        self._video_pipeline.add_stage('demuxer', VideoPipelineStagePayloadType.Frame)
        self._video_pipeline.add_stage(
            'output-queue', VideoPipelineStagePayloadType.Frame
        )
        sink_peer_pad: Gst.Pad = demuxer.get_static_pad('sink').get_peer()
        add_move_batch_as_is_pad_probe(
            sink_peer_pad,
            self._video_pipeline,
            'update-frame-meta',
        )
        sink_peer_pad.add_probe(Gst.PadProbeType.BUFFER, self.update_frame_meta)
        add_move_batch_to_frames_pad_probe(
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
            return Gst.PadProbeReturn.PASS
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
            return Gst.PadProbeReturn.PASS

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


class PipelineIsNotRunningError(Exception):
    """Pipeline is not running."""


def _init_telemetry(module_name: str, telemetry: TelemetryParameters):
    """Initialize telemetry provider."""

    provider_params = telemetry.provider_params or {}
    if telemetry.provider == 'jaeger':
        service_name = provider_params.get('service_name', module_name)
        try:
            endpoint = provider_params['endpoint']
        except KeyError:
            raise ValueError(
                'Jaeger endpoint is not specified. Please specify it in the config file.'
            )
        init_jaeger_tracer(service_name, endpoint)

    elif telemetry.provider is not None:
        raise ValueError(f'Unknown telemetry provider: {telemetry.provider}')
