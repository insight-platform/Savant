"""Overlay for meta-merge visualization - draws bounding boxes for detected objects."""

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist, Position


class Overlay(NvDsDrawFunc):
    """Draws bounding boxes and labels for all detected objects."""

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draw bounding boxes for each object in the frame."""
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                continue

            # Draw bounding box
            artist.add_bbox(obj_meta.bbox, 2, (0, 255, 0, 255))

            # Draw label
            label = f"{obj_meta.label}"
            if obj_meta.confidence is not None:
                label += f" {obj_meta.confidence:.2f}"
            artist.add_text(
                label,
                (int(obj_meta.bbox.left), int(obj_meta.bbox.top) - 5),
                0.5,
                1,
                anchor_point_type=Position.LEFT_BOTTOM,
            )
