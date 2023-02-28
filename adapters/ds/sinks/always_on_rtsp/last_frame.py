from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2


@dataclass
class LastFrame:
    timestamp: datetime
    frame: Optional[cv2.cuda.GpuMat] = None
