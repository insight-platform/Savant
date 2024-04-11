"""Ingress / egress filters module."""

from abc import abstractmethod

from savant_rs.primitives import VideoFrame

from savant.base.pyfunc import BasePyFuncCallableImpl


class BaseFrameFilter(BasePyFuncCallableImpl):
    """Frame filter interface."""

    @abstractmethod
    def __call__(self, video_frame: VideoFrame) -> bool:
        """Filter for frames.

        :param video_frame: Video frame.
        :return: Whether to include the frame (True) or skip it (False).
        """


class DefaultEgressFilter(BaseFrameFilter):
    """Default egress filter, allows all frame to pass."""

    def __call__(self, video_frame: VideoFrame) -> bool:
        """Filters output frames.

        :param video_frame: Video frame.
        :return: Whether to put the frame into the output queue (True) or skip it (False).
        """

        return True


class DefaultIngressFilter(BaseFrameFilter):
    """Default ingress filter, filters out frames with no video data."""

    def __call__(self, video_frame: VideoFrame) -> bool:
        """Filters input frames.

        :param video_frame: Video frame.
        :return: Whether to accept the frame into the pipeline (True) or skip it (False).
        """

        return not video_frame.content.is_none()
