"""Base implementation of user-defined PyFunc class."""
import numpy as np
import pyds
from savant.base.pyfunc import BasePyFuncPlugin
from savant.deepstream.utils import nvds_frame_meta_iterator
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.gstreamer import Gst  # noqa: F401


class NvDsPyFuncPlugin(BasePyFuncPlugin):
    """DeepStream PyFunc plugin base class.

    PyFunc implementations are defined in and instantiated by a
    :py:class:`.PyFunc` structure.
    """

    def process_buffer(self, buffer: Gst.Buffer):
        """Process gstreamer buffer directly. Throws an exception if fatal
        error has occurred.

        Default implementation calls :py:func:`~NvDsPyFuncPlugin.process_frame_meta`
        and :py:func:`~NvDsPyFuncPlugin.process_frame` for each frame in a batch.

        :param buffer: Gstreamer buffer.
        """
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            frame = pyds.get_nvds_buf_surface(hash(buffer), nvds_frame_meta.batch_id)
            frame_meta = NvDsFrameMeta(frame_meta=nvds_frame_meta)
            self.process_frame_meta(frame_meta)
            self.process_frame(frame_meta, frame)

    def process_frame_meta(self, frame_meta: NvDsFrameMeta):
        """Process frame metadata. Throws an exception if fatal error has
        occurred.

        :param frame_meta: Frame metadata for a frame in a batch.
        """

    def process_frame(self, frame_meta: NvDsFrameMeta, frame: np.ndarray):
        """Process frame metadata and frame image. Throws an exception if fatal
        error has occurred.

        :param frame_meta: Frame metadata for a frame in a batch.
        :param frame: Current frame in RGBA format, represented as a numpy array.
        """
