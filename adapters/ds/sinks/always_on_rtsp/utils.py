from typing import Tuple, Union

import cv2
import numpy as np

from savant.gstreamer import Gst

Frame = Union[cv2.cuda.GpuMat, np.ndarray]
"""Type alias for a frame content."""


def nvidia_runtime_is_available() -> bool:
    """Check if Nvidia runtime is available."""

    return Gst.ElementFactory.find('nvv4l2h264enc') is not None


def get_frame_resolution(frame: Frame) -> Tuple[int, int]:
    """Get frame resolution."""

    if isinstance(frame, cv2.cuda.GpuMat):
        return frame.size()
    return frame.shape[1], frame.shape[0]
