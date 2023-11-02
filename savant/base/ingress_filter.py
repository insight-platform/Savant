"""Base ingress filter."""
from abc import abstractmethod

from savant_rs.primitives import VideoFrame
from savant.base.pyfunc import BasePyFuncCallableImpl


class BaseIngressFilter(BasePyFuncCallableImpl):


    @abstractmethod
    def __call__(self, video_frame: VideoFrame) -> bool:
        """Filters input frames.

        :param video_frame: Video frame.
        :return: Whether to accept the frame into the pipeline.
        """


class DefaultIngressFilter(BaseIngressFilter):


    def __call__(self, video_frame: VideoFrame) -> bool:
        """Filters input frames.

        :param video_frame: Video frame.
        :return: Whether to accept the frame into the pipeline.
        """

        return not video_frame.content.is_none()
