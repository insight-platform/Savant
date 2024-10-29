from abc import ABC, abstractmethod
from typing import Optional

from savant_rs.primitives import VideoFrame

from savant.config.schema import FrameParameters
from savant.utils.logging import get_logger
from savant.utils.source_info import SourceShape


class BaseSourceShaper(ABC):
    """Base class to define a source shape.

    :param geometry_base: Base value for frame parameters. All frame parameters must be divisible by this value.
    :param kwargs: Custom keyword arguments.
        They will be available inside the class instance,
        as fields with the argument name.
    """

    def __init__(self, geometry_base: int, **kwargs):
        self.geometry_base = geometry_base
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.logger = get_logger(self.__module__)

    @abstractmethod
    def __call__(
        self,
        source_id: str,
        width: int,
        height: int,
        frame_meta: VideoFrame,
    ) -> Optional[SourceShape]:
        """Get the source shape for the given source.

        :param source_id: Source ID
        :param width: Source width
        :param height: Source height
        :param frame_meta: Metadata of the first frame in the source.
        :return: Shape of the source or None. When None is returned, the source shape will not be modified.

        .. note:: The source shape should be divisible by the geometry base.
        """
        pass


class DefaultSourceShaper(BaseSourceShaper):
    """Default source shaper.

    Uses the frame parameters from configuration to determine the source shape.
    """

    frame_params: FrameParameters

    def __call__(
        self,
        source_id: str,
        width: int,
        height: int,
        frame_meta: VideoFrame,
    ) -> Optional[SourceShape]:
        if self.frame_params.width and self.frame_params.height:
            return SourceShape(
                width=self.frame_params.width,
                height=self.frame_params.height,
                padding=self.frame_params.padding,
            )
