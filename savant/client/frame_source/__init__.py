from abc import ABC, abstractmethod
from typing import Tuple

from savant_rs.primitives import VideoFrame, VideoFrameUpdate


class FrameSource(ABC):
    """Interface for frame sources."""

    @abstractmethod
    def with_update(self, update: VideoFrameUpdate) -> 'FrameSource':
        """Apply an update to a frame."""
        pass

    @abstractmethod
    def build_frame(self) -> Tuple[VideoFrame, bytes]:
        """Build a frame.

        :return: A tuple of a frame metadata and its content.
        """
        pass
