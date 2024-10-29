from typing import List, Tuple

from savant_rs.primitives.geometry import BBox

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist


class Overlay(NvDsDrawFunc):
    """Colors padding in the frame."""

    def __init__(self, color: List[int], **kwargs):
        super().__init__(**kwargs)
        self.color: Tuple[int, int, int, int] = tuple(color)

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        self.logger.debug(
            'Processing frame %s/%s',
            frame_meta.source_id,
            frame_meta.pts,
        )
        if not frame_meta.video_frame.transformations:
            self.logger.debug(
                'No transformations found in frame %s/%s',
                frame_meta.source_id,
                frame_meta.pts,
            )
            return
        last_transformation = frame_meta.video_frame.transformations[-1]
        if not last_transformation.is_padding:
            self.logger.debug(
                'Last transformation in frame %s/%s is not padding',
                frame_meta.source_id,
                frame_meta.pts,
            )
            return

        left, top, right, bottom = last_transformation.as_padding
        self.logger.debug(
            'Padding in frame %s/%s: left=%s, top=%s, right=%s, bottom=%s',
            frame_meta.source_id,
            frame_meta.pts,
            left,
            top,
            right,
            bottom,
        )
        if left > 0:
            self.logger.debug(
                'Adding left padding to frame %s/%s',
                frame_meta.source_id,
                frame_meta.pts,
            )
            artist.add_bbox(
                BBox(left // 2, artist.height // 2, left, artist.height),
                border_width=0,
                border_color=self.color,
                bg_color=self.color,
            )
        if top > 0:
            self.logger.debug(
                'Adding top padding to frame %s/%s',
                frame_meta.source_id,
                frame_meta.pts,
            )
            artist.add_bbox(
                BBox(artist.width // 2, top // 2, artist.width, top),
                border_width=0,
                border_color=self.color,
                bg_color=self.color,
            )
        if right > 0:
            self.logger.debug(
                'Adding right padding to frame %s/%s',
                frame_meta.source_id,
                frame_meta.pts,
            )
            artist.add_bbox(
                BBox(
                    artist.width - right // 2, artist.height // 2, right, artist.height
                ),
                border_width=0,
                border_color=self.color,
                bg_color=self.color,
            )
        if bottom > 0:
            self.logger.debug(
                'Adding bottom padding to frame %s/%s',
                frame_meta.source_id,
                frame_meta.pts,
            )
            artist.add_bbox(
                BBox(
                    artist.width // 2, artist.height - bottom // 2, artist.width, bottom
                ),
                border_width=0,
                border_color=self.color,
                bg_color=self.color,
            )
        self.logger.debug('Frame %s/%s processed', frame_meta.source_id, frame_meta.pts)
