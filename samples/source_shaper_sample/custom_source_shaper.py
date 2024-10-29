from typing import Dict, Optional

from savant_rs.primitives import VideoFrame

from savant.base.source_shaper import BaseSourceShaper
from savant.config.schema import FramePadding
from savant.utils.source_info import SourceShape


class CustomSourceShaper(BaseSourceShaper):
    def __init__(self, geometry_base: int, sources: Dict[str, Dict], **kwargs):
        super().__init__(geometry_base, **kwargs)
        self.sources = sources

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

        source = self.sources.get(source_id)
        if source is None:
            self.logger.info('No source shape for source %s', source_id)
            return None

        padding = FramePadding(**source['padding']) if source.get('padding') else None
        shape = SourceShape(
            width=source['width'],
            height=source['height'],
            padding=padding,
        )

        self.logger.info(
            'Source %s with original resolution %sx%s shaped to %s. Original codec: %s, transformations: %s.',
            source_id,
            width,
            height,
            shape,
            frame_meta.codec,
            frame_meta.transformations,
        )

        return shape
