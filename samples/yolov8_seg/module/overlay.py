"""Custom DrawFunc implementation."""
import numpy as np
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist


class Overlay(NvDsDrawFunc):
    """Custom implementation of PyFunc for drawing on frame."""

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws on frame using the artist and the frame's metadata.

        :param frame_meta: Frame metadata.
        :param artist: Artist to draw on the frame.
        """
        # need to init artist.overlay!
        super().draw_on_frame(frame_meta, artist)

        mask_color = np.array([0, 255, 0, 64], dtype=np.uint8)
        bg_color = np.array([0, 0, 0, 0], dtype=np.uint8)

        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                continue

            mask_attr = obj_meta.get_attr_meta('yolov8_seg', 'mask')
            if not mask_attr:
                continue

            mask = mask_attr.value
            mask_overlay = np.where(mask[..., None], mask_color, bg_color)
            left, top = int(obj_meta.bbox.left), int(obj_meta.bbox.top)
            artist.overlay[
                top : top + mask_overlay.shape[0],
                left : left + mask_overlay.shape[1],
            ] |= mask_overlay
