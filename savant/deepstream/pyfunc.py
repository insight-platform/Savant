"""Base implementation of user-defined PyFunc class."""
import pyds
import cv2
from savant.base.pyfunc import BasePyFuncPlugin
from savant.deepstream.utils import (
    nvds_frame_meta_iterator,
    GST_NVEVENT_STREAM_EOS,
    gst_nvevent_parse_stream_eos,
)
from savant.deepstream.meta.frame import NvDsFrameMeta
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
        self.batch_streams = []

    def on_sink_event(self, event: Gst.Event):
        """Add stream event callbacks."""
        if event.type == GST_NVEVENT_STREAM_EOS:
            pad_idx = gst_nvevent_parse_stream_eos(event)
            if pad_idx is not None:
                source_id = self._sources.get_id_by_pad_index(pad_idx)
                self.on_source_eos(source_id)

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        # self.logger.debug('Got GST_NVEVENT_STREAM_EOS for source %s.', source_id)

    def get_cuda_stream(self):
        """"""
        stream = cv2.cuda.Stream()
        self.batch_streams.append(stream)
        return stream

    def process_buffer(self, buffer: Gst.Buffer):
        """Process gstreamer buffer directly. Throws an exception if fatal
        error has occurred.

        Default implementation calls :py:func:`~NvDsPyFuncPlugin.process_frame_meta`
        and :py:func:`~NvDsPyFuncPlugin.process_frame` for each frame in a batch.

        :param buffer: Gstreamer buffer.
        """
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            frame_meta = NvDsFrameMeta(frame_meta=nvds_frame_meta)
            self.process_frame(buffer, frame_meta)

        for stream in self.batch_streams:
            stream.waitForCompletion()
        self.batch_streams = []

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process gstreamer buffer and frame metadata. Throws an exception if fatal
        error has occurred.

        Use `savant.deepstream.utils.get_nvds_buf_surface` to get a frame image.

        :param buffer: Gstreamer buffer.
        :param frame_meta: Frame metadata for a frame in a batch.
        """
