from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from adapters.ds.sinks.always_on_rtsp.utils import Frame


@dataclass
class LastFrame:
    timestamp: datetime
    frame: Optional[Frame] = None
