from abc import ABC, abstractmethod
from typing import Tuple, TypeVar

from savant_rs.primitives import VideoFrame, VideoFrameUpdate

T = TypeVar('T', bound='FrameSource')


class FrameSource(ABC):
    """Interface for frame sources."""

    @abstractmethod
    def with_update(self: T, update: VideoFrameUpdate) -> T:
        """Apply an update to a frame."""

    @abstractmethod
    def build_frame(self) -> Tuple[VideoFrame, bytes]:
        """Build a frame.

        :return: A tuple of a frame metadata and its content.
        """
