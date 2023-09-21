"""Base implementation of user-defined PyFunc class."""
from typing import Dict, Optional

import cv2
import pyds
from pygstsavantframemeta import (
    gst_buffer_get_savant_batch_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)
from savant_rs.pipeline2 import VideoPipeline

from savant.base.pyfunc import BasePyFuncPlugin
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.utils import (
    GST_NVEVENT_PAD_ADDED,
    GST_NVEVENT_PAD_DELETED,
    GST_NVEVENT_STREAM_EOS,
    gst_nvevent_parse_pad_added,
    gst_nvevent_parse_pad_deleted,
    gst_nvevent_parse_stream_eos,
    nvds_frame_meta_iterator,
)
from savant.gstreamer import Gst  # noqa: F401
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
        self._last_nvevent_seqnum: Dict[int, Dict[int, int]] = {
            event_type: {}
            for event_type in [
                GST_NVEVENT_PAD_ADDED,
                GST_NVEVENT_PAD_DELETED,
                GST_NVEVENT_STREAM_EOS,
            ]
        }
        self.frame_streams = {}

    def on_start(self) -> bool:
        """Do on plugin start."""
        self._video_pipeline = self.gst_element.get_property('pipeline')
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
        """Get a CUDA stream that can be used to
        asynchronously process a frame in a batch.
        All frame CUDA streams will be waited for at the end of batch processing.
        """
        self.logger.debug(
            'Getting CUDA stream for frame with batch_id=%d', frame_meta.batch_id
        )
        if frame_meta.batch_id not in self.frame_streams:
            self.logger.debug(
                'No existing CUDA stream for frame with batch_id=%d, init new',
                frame_meta.batch_id,
            )
            self.frame_streams[frame_meta.batch_id] = cv2.cuda.Stream()

        return self.frame_streams[frame_meta.batch_id]

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

        for stream in self.frame_streams.values():
            stream.waitForCompletion()
        self.frame_streams.clear()

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
