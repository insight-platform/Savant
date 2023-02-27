"""DeepStream surface utils."""
from contextlib import contextmanager
from typing import ContextManager
import pyds
import numpy as np
from savant.gstreamer import Gst  # noqa:F401


@contextmanager
def get_nvds_buf_surface(
    buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
) -> ContextManager[np.ndarray]:
    """This function returns the frame in NumPy format. Only RGBA format is
    supported. For x86_64, only unified memory is supported. For Jetson, the
    buffer is mapped to CPU memory. Changes to the frame image will be
    preserved and seen in downstream elements, with the following restrictions.

    1. No change to image color format or resolution

    2. No transpose operation on the array.

    The function automatically unmaps NvDsBufSurface to prevent memory leak.
    See https://github.com/insight-platform/Savant/issues/25 for the details.

    Usage example:

    .. code-block:: python

        with get_nvds_buf_surface(buffer, nvds_frame_meta) as np_frame:
            frame_bytes = np_frame.tobytes()

    :param buffer: Gstreamer buffer which contains NvBufSurface.
    :param nvds_frame_meta: NvDs frame metadata which contains frame info.
    :return: Frame in NumPy format.
    """

    try:
        yield pyds.get_nvds_buf_surface(hash(buffer), nvds_frame_meta.batch_id)
    finally:
        pyds.unmap_nvds_buf_surface(hash(buffer), nvds_frame_meta.batch_id)
