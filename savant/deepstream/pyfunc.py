"""Base implementation of user-defined PyFunc class."""

from typing import Any, Dict, Optional

import cv2
import pyds
from pygstsavantframemeta import (
    gst_buffer_get_savant_batch_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)
from savant_rs.pipeline2 import VideoPipeline

from savant.api.constants import DEFAULT_FRAMERATE
from savant.base.pyfunc import BasePyFuncPlugin
from savant.deepstream.auxiliary_stream import (
    AuxiliaryStream,
    AuxiliaryStreamInternal,
    AuxiliaryStreamRegistry,
)
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.utils.event import (
    GST_NVEVENT_PAD_ADDED,
    GST_NVEVENT_PAD_DELETED,
    GST_NVEVENT_STREAM_EOS,
    gst_nvevent_parse_pad_added,
    gst_nvevent_parse_pad_deleted,
    gst_nvevent_parse_stream_eos,
)
from savant.deepstream.utils.iterator import nvds_frame_meta_iterator
from savant.gstreamer import Gst  # noqa: F401
from savant.metrics.base import BaseMetricsExporter
from savant.metrics.registry import MetricsRegistry
from savant.utils.platform import is_aarch64
from savant.utils.source_info import SourceInfoRegistry


class NvDsPyFuncPlugin(BasePyFuncPlugin):
    """DeepStream PyFunc plugin base class.

    PyFunc implementations are defined in and instantiated by a
    :py:class:`.PyFunc` structure.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sources = SourceInfoRegistry()
        self._video_pipeline: Optional[VideoPipeline] = None
        self._metrics_exporter: Optional[BaseMetricsExporter] = None
        self._metrics_registry: Optional[MetricsRegistry] = None
        self._last_nvevent_seqnum: Dict[int, Dict[int, int]] = {
            event_type: {}
            for event_type in [
                GST_NVEVENT_PAD_ADDED,
                GST_NVEVENT_PAD_DELETED,
                GST_NVEVENT_STREAM_EOS,
            ]
        }
        self._stream_pool_size = None
        self._stream_pool = []
        # TODO: make AuxiliaryStreamRegistry global?
        self._auxiliary_streams = AuxiliaryStreamRegistry()

    def on_start(self) -> bool:
        """Do on plugin start."""
        self._video_pipeline = self.gst_element.get_property('pipeline')
        self._metrics_exporter = self.gst_element.get_property('metrics-exporter')
        self._metrics_registry = MetricsRegistry(self._metrics_exporter)
        # the prop is set to pipeline batch size during init
        self._stream_pool_size = self.gst_element.get_property('stream-pool-size')
        return True

    def on_event(self, event: Gst.Event):
        """Add stream event callbacks."""
        # GST_NVEVENT_STREAM_START is not sent,
        # so use GST_NVEVENT_PAD_ADDED/DELETED
        if event.type == GST_NVEVENT_PAD_ADDED:
            pad_idx = gst_nvevent_parse_pad_added(event)
            if self._is_processed(event, pad_idx):
                return
            source_id = self._sources.get_id_by_pad_index(pad_idx)
            self.on_source_add(source_id)

        elif event.type == GST_NVEVENT_PAD_DELETED:
            pad_idx = gst_nvevent_parse_pad_deleted(event)
            if self._is_processed(event, pad_idx):
                return
            source_id = self._sources.get_id_by_pad_index(pad_idx)
            self.on_source_delete(source_id)

        elif event.type == GST_NVEVENT_STREAM_EOS:
            pad_idx = gst_nvevent_parse_stream_eos(event)
            if self._is_processed(event, pad_idx):
                return
            source_id = self._sources.get_id_by_pad_index(pad_idx)
            self.on_source_eos(source_id)

    def on_source_add(self, source_id: str):
        """On source add event callback."""
        # self.logger.debug('Source %s added.', source_id)

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        # self.logger.debug('Source %s EOS.', source_id)

    def on_source_delete(self, source_id: str):
        """On source delete event callback."""
        # self.logger.debug('Source %s deleted.', source_id)

    def get_cuda_stream(self, frame_meta: NvDsFrameMeta):
        """Get a CUDA stream that can be used to asynchronously process
        a frame in a batch.

        All frame CUDA streams will be waited for at the end of batch processing.
        """
        if not self._stream_pool:
            self.logger.debug(
                'Creating CUDA stream pool of size %d.', self._stream_pool_size
            )
            self._stream_pool = [
                cv2.cuda.Stream() for _ in range(self._stream_pool_size)
            ]
        self.logger.debug('Using CUDA stream %d.', frame_meta.batch_id)
        return self._stream_pool[frame_meta.batch_id]

    def process_buffer(self, buffer: Gst.Buffer):
        """Process gstreamer buffer directly. Throws an exception if fatal
        error has occurred.

        Default implementation calls :py:func:`~NvDsPyFuncPlugin.process_frame_meta`
        and :py:func:`~NvDsPyFuncPlugin.process_frame` for each frame in a batch.

        :param buffer: Gstreamer buffer.
        """
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        savant_batch_meta = gst_buffer_get_savant_batch_meta(buffer)
        if savant_batch_meta is None:
            self.logger.warning(
                'Failed to process batch at buffer %s. '
                'Batch has no Savant Frame Meta.',
                buffer.pts,
            )
            return

        batch_id = savant_batch_meta.idx

        self.logger.debug(
            'Processing batch id=%d, with %d frames',
            id(nvds_batch_meta),
            nvds_batch_meta.num_frames_in_batch,
        )
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            if savant_frame_meta is None:
                self.logger.warning(
                    'Failed to process frame %s at buffer %s. '
                    'Frame has no Savant Frame Meta.',
                    nvds_frame_meta.buf_pts,
                    buffer.pts,
                )
                continue

            frame_id = savant_frame_meta.idx
            video_frame, video_frame_span = self._video_pipeline.get_batched_frame(
                batch_id,
                frame_id,
            )
            with video_frame_span.nested_span('process-frame') as telemetry_span:
                with NvDsFrameMeta(
                    nvds_frame_meta,
                    video_frame,
                    telemetry_span,
                ) as frame_meta:
                    self.process_frame(buffer, frame_meta)

        for stream in self._stream_pool:
            stream.waitForCompletion()

        self._auxiliary_streams.flush()

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process gstreamer buffer and frame metadata. Throws an exception if fatal
        error has occurred.

        Use `savant.deepstream.utils.get_nvds_buf_surface` to get a frame image.

        :param buffer: Gstreamer buffer.
        :param frame_meta: Frame metadata for a frame in a batch.
        """

    def _is_processed(self, event: Gst.Event, pad_idx: int) -> bool:
        """Check if event has already been processed."""

        if event.get_seqnum() <= self._last_nvevent_seqnum[event.type].get(pad_idx, -1):
            return True
        self._last_nvevent_seqnum[event.type][pad_idx] = event.get_seqnum()
        return False

    def get_runtime_metrics(self, n: int):
        """Get last runtime metrics."""

        return self._video_pipeline.get_stat_records(n)

    @property
    def metrics(self) -> MetricsRegistry:
        """Get metrics registry.

        Usage example:

        .. code-block:: python

            from savant.metrics import Counter
            self.metrics['frames_per_source'] = Counter(
                name='frames_per_source',
                description='Number of processed frames per source',
                labelnames=('source_id',),
            )
            ...
            self.metrics['frames_per_source'].inc(labels=('camera-1',))
        """

        return self._metrics_registry

    def auxiliary_stream(
        self,
        source_id: str,
        width: int,
        height: int,
        codec_params: Dict[str, Any],
        framerate: str = DEFAULT_FRAMERATE,
    ) -> AuxiliaryStream:
        """Create an auxiliary stream.

        Frames from auxiliary streams are encoded and sent to sink. They are
        not sent to downstream pipeline elements. This can be useful to send
        video in different resolutions, codecs, etc.

        Usage example:

        .. code-block:: python

            aux_stream = self.auxiliary_stream(
                source_id='aux-source',
                width=640,
                height=480,
                codec_params={'codec': 'h264'},
            )

            ...

            cuda_stream = self.get_cuda_stream(frame_meta)
            aux_frame, aux_buffer = aux_stream.create_frame(pts=pts)
            with nvds_to_gpu_mat(aux_buffer, batch_id=0) as aux_mat:
                cv2.cuda.resize(
                    src=orig_frame_mat,
                    dst=aux_mat,
                    dsize=(640, 480),
                    stream=cuda_stream,
                )

        :param source_id: Source ID. Must be unique.
        :param width: Frame width.
        :param height: Frame height.
        :param codec_params: Codec parameters. Only "h264" and "hevc" codecs
            are supported.
        :param framerate: Framerate. Default is 30/1.
        """

        if is_aarch64():
            raise RuntimeError('Auxiliary streams are not supported on aarch64')

        if self._auxiliary_streams.get_stream(source_id) is not None:
            raise RuntimeError(f'Auxiliary stream {source_id} already exists')

        self.logger.info('Requesting pad for source %s', source_id)
        pad: Gst.Pad = self.gst_element.request_pad_simple('aux_src_%u')
        if pad is None:
            raise RuntimeError(f'Failed to request pad for source {source_id}')
        self.logger.info('Got pad %s for source %s', pad.get_name(), source_id)

        aux_stream_internal = AuxiliaryStreamInternal(
            source_id=source_id,
            sources=self._sources,
            width=width,
            height=height,
            framerate=framerate,
            codec_params=codec_params,
            video_pipeline=self._video_pipeline,
            # stage_name=self.gst_element.get_name(),
            stage_name='source',
            gst_pipeline=self.gst_element.get_property('gst-pipeline'),
            pad=pad,
        )
        aux_stream = AuxiliaryStream(aux_stream_internal, self._auxiliary_streams)
        self._auxiliary_streams.add_stream(aux_stream_internal)

        return aux_stream
