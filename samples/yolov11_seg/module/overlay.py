"""Custom DrawFunc implementation."""

from typing import List

import numpy as np

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.utils.artist import Artist


class Overlay(NvDsDrawFunc):
    """Custom implementation of PyFunc for drawing on frame."""

    def __init__(self, mask_color: List[int], bg_color: List[int], **kwargs):
        super().__init__(**kwargs)
        self.mask_color = np.array(mask_color, dtype=np.uint8)
        self.bg_color = np.array(bg_color, dtype=np.uint8)

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        super().draw_on_frame(frame_meta, artist)

        for obj_meta in frame_meta.objects:
            if obj_meta.label == 'person':
                mask_attr = obj_meta.get_attr_meta('segmenter', 'mask')
                if not mask_attr:
                    continue

                bbox = obj_meta.bbox.as_ltrb_int()

                mask_overlay = np.where(
                    mask_attr.value[..., None], self.mask_color, self.bg_color
                )[0 : bbox[3] - bbox[1], 0 : bbox[2] - bbox[0]]

                # if any shape dimension is 0, skip
                if any(dim == 0 for dim in mask_overlay.shape):
                    continue

                artist.add_graphic(
                    img=mask_overlay,
                    origin=(bbox[0], bbox[1]),
                )
