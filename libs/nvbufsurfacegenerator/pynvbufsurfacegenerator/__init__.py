import gi

from . import pynvbufsurfacegenerator

gi.require_version('Gst', '1.0')
from gi.repository import Gst


class NvBufSurfaceGenerator:
    """Generates GStreamer buffers with NvBufSurface memory allocated.

    :param caps: Caps for the generated buffers.
    :param gpu_id: ID of the GPU to allocate NvBufSurface.
    :param mem_type: Memory type for the NvBufSurface.
    """

    def __init__(self, caps: Gst.Caps, gpu_id: int, mem_type: int):
        self._nested = pynvbufsurfacegenerator.NvBufSurfaceGenerator(
            hash(caps), gpu_id, mem_type
        )

    def create_surface(self, buffer: Gst.Buffer):
        """Create a new NvBufSurface and attach it to the given buffer.

        :param buffer: Buffer to attach the NvBufSurface to.
        """
        return self._nested.create_surface(hash(buffer))
