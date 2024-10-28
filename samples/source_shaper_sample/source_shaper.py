from typing import Optional

from savant_rs.primitives import VideoFrame

from savant.base.source_shaper import BaseSourceShaper
from savant.utils.source_info import SourceShape


class SourceShaper(BaseSourceShaper):
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
        """

        # TODO: use the frame metadata in some way
        self.logger.info('Getting source shape for %s: %sx%s', source_id, width, height)

        return SourceShape(width=1280, height=720)
