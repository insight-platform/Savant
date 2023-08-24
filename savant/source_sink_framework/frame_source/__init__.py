from abc import ABC, abstractmethod
from typing import Tuple

from savant_rs.primitives import VideoFrame, VideoFrameUpdate


class FrameSource(ABC):
    @abstractmethod
    def with_update(self, update: VideoFrameUpdate) -> 'FrameSource':
        pass

    @abstractmethod
    def build_frame(self) -> Tuple[VideoFrame, bytes]:
        pass
