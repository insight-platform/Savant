import functools
from typing import Union

import cv2
import numpy as np

from savant.gstreamer import Gst

Frame = Union[cv2.cuda.GpuMat, np.ndarray, bytes]
"""Type alias for a frame content."""


@functools.lru_cache
def nvidia_runtime_is_available() -> bool:
    """Check if Nvidia runtime is available."""

    return Gst.ElementFactory.find('nvv4l2h264enc') is not None
