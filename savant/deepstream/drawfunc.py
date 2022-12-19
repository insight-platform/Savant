"""Default implementation PyFunc for drawing on frame."""

import numpy as np
import pyds
from cairo import Context, Format, ImageSurface
from savant.deepstream.base_drawfunc import BaseNvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.meta.bbox import BBox, RBBox
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.utils.artist import Position, Artist


class NvDsDrawFunc(BaseNvDsDrawFunc):
    """Default implementation of PyFunc for drawing on frame.

    PyFunc implementations are defined in and instantiated by a
    :py:class:`.PyFunc` structure.
    """

    def __call__(self, nvds_frame_meta: pyds.NvDsFrameMeta, frame: np.ndarray):
        frame_height, frame_width, _ = frame.shape
        surface = ImageSurface.create_for_data(
            frame, Format.ARGB32, frame_width, frame_height
        )
        artist = Artist(Context(surface))
        frame_meta = NvDsFrameMeta(frame_meta=nvds_frame_meta)
        self.draw_on_frame(frame_meta, artist)
        surface.flush()
        surface.finish()

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws bounding boxes and labels for all objects in the frame.

        :param frame_meta: Frame metadata for a frame in a batch.
        :param artist: Cairo context drawer to drawing primitives and directly on frame.
        """
        for obj_meta in frame_meta.objects:
            if not obj_meta.element_name and obj_meta.label == 'frame':
                continue

            artist.add_bbox(bbox=obj_meta.bbox)

            label = obj_meta.label
            if obj_meta.track_id != UNTRACKED_OBJECT_ID:
                label += f' #{obj_meta.track_id}'

            if isinstance(obj_meta.bbox, BBox):
                artist.add_text(
                    text=label,
                    anchor_x=int(obj_meta.bbox.left),
                    anchor_y=int(obj_meta.bbox.top),
                    bg_color=(0.0, 0.0, 0.0),
                    anchor_point=Position.LEFT_TOP,
                )

            elif isinstance(obj_meta.bbox, RBBox):
                artist.add_text(
                    text=label,
                    anchor_x=int(obj_meta.bbox.x_center),
                    anchor_y=int(obj_meta.bbox.y_center),
                    bg_color=(0.0, 0.0, 0.0),
                    anchor_point=Position.CENTER,
                )
