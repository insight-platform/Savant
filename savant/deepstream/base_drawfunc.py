"""Base PyFunc for drawing on frame."""

from abc import abstractmethod

import pyds

from savant.base.pyfunc import BasePyFuncCallableImpl
from savant.gstreamer import Gst  # noqa: F401


class BaseNvDsDrawFunc(BasePyFuncCallableImpl):
    """Base PyFunc for drawing on frame.

    PyFunc implementations are defined in and instantiated by a
    :py:class:`.PyFunc` structure.
    """

    @abstractmethod
    def __call__(self, nvds_frame_meta: pyds.NvDsFrameMeta, buffer: Gst.Buffer):
        """Processes gstreamer buffer with Deepstream batch structure. Draws
        Deepstream metadata objects for each frame in the batch. Throws an
        exception if fatal error has occurred.

        :param nvds_frame_meta: NvDs metadata for a frame.
        :param buffer: Gstreamer buffer.
        """

    @abstractmethod
    def finalize(self):
        """Finalize batch processing.
        """
