from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import cv2
import numpy as np

Frame = Union[cv2.cuda.GpuMat, np.ndarray, bytes]
"""Type alias for a frame content."""


@dataclass
class LastFrame:
    timestamp: datetime
    width: int
    height: int
    content: Frame


@dataclass
class LastFrameRef:
    frame: Optional[LastFrame] = None
