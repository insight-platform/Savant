"""PyGroup plugin implementation."""

from typing import List, Optional

import pyds
from pygstsavantframemeta import (
    gst_buffer_get_savant_batch_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)
from savant_rs.pipeline2 import VideoPipeline

from savant.base.pyfunc import BasePyFuncPlugin, PyFunc
from savant.gstreamer import Gst  # noqa: F401

from .meta.frame import NvDsFrameMeta
from .utils.iterator import nvds_frame_meta_iterator


class NvDsPyGroupPlugin(BasePyFuncPlugin):
    """Groups multiple NvDsPyFuncPlugin instances and executes them
    sequentially on each buffer.

    :param pyfuncs: List of PyFunc wrappers for the sub-pyfuncs.
    :param pyfunc_span_names: Telemetry span name for each sub-pyfunc.
    """

    def __init__(
        self,
        pyfuncs: List[PyFunc],
        pyfunc_span_names: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pyfuncs = pyfuncs
        self._pyfunc_span_names = pyfunc_span_names
        self._video_pipeline: Optional[VideoPipeline] = None

    def on_start(self) -> bool:
        self._video_pipeline = self.gst_element.get_property('pipeline')
        for pyfunc in self._pyfuncs:
            pyfunc.instance.gst_element = self.gst_element
            if not pyfunc.instance.on_start():
                return False
        return True

    def on_stop(self) -> bool:
        for pyfunc in self._pyfuncs:
            if not pyfunc.instance.on_stop():
                return False
        return True

    def on_event(self, event: Gst.Event):
        """Add stream event callback."""
        for pyfunc in self._pyfuncs:
            pyfunc.instance.on_event(event)

    def process_buffer(self, buffer: Gst.Buffer):
        """Process gstreamer buffer.

        Iterates over frames in the batch.  For each frame, calls every
        sub-pyfunc's ``process_frame`` inside an individually-named
        telemetry span.

        :param buffer: Gstreamer buffer.
        """

        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        savant_batch_meta = gst_buffer_get_savant_batch_meta(buffer)
        if savant_batch_meta is None:
            self.logger.warning(
                'Failed to process batch at buffer %s. Batch has no Savant Frame Meta.',
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
            with video_frame_span.nested_span('process-frame') as pygroup_span:
                for pyfunc, span_name in zip(self._pyfuncs, self._pyfunc_span_names):
                    with pygroup_span.nested_span(span_name) as pyfunc_span:
                        with NvDsFrameMeta(
                            nvds_frame_meta,
                            video_frame,
                            pyfunc_span,
                        ) as frame_meta:
                            pyfunc.instance.process_frame(buffer, frame_meta)

        for pyfunc in self._pyfuncs:
            for stream in pyfunc.instance._stream_pool:
                stream.waitForCompletion()
            pyfunc.instance._auxiliary_streams.flush()
