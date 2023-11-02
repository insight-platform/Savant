"""Base egress filter."""
from abc import abstractmethod

from savant_rs.primitives import VideoFrame
from savant.base.pyfunc import BasePyFuncCallableImpl


class BaseEgressFilter(BasePyFuncCallableImpl):


    @abstractmethod
    def __call__(self, video_frame: VideoFrame) -> bool:
        """Filters output frames.

        :param video_frame: Video frame.
        :return: Whether to put the frame into the output queue.
        """


class DefaultEgressFilter(BaseEgressFilter):


    def __call__(self, video_frame: VideoFrame) -> bool:
        """Filters output frames.

        :param video_frame: Video frame.
        :return: Whether to put the frame into the output queue.
        """

        return True
