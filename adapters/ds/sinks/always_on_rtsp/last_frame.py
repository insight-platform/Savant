from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from adapters.ds.sinks.always_on_rtsp.utils import Frame


@dataclass
class LastFrame:
    timestamp: datetime
    width: int
    height: int
    content: Frame


@dataclass
class LastFrameRef:
    frame: Optional[LastFrame] = None
