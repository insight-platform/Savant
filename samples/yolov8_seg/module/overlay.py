"""Custom DrawFunc implementation."""
import numpy as np

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import alpha_comp, draw_rect, nvds_to_gpu_mat
from savant.gstreamer import Gst  # noqa: F401


class Overlay(NvDsDrawFunc):
    """Custom implementation of PyFunc for drawing on frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bbox_color = (0, 255, 0, 255)
        self.mask_color = np.array([0, 255, 0, 64], dtype=np.uint8)
        self.bg_color = np.array([0, 0, 0, 0], dtype=np.uint8)

    def draw(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        stream = self.get_cuda_stream()
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            for obj_meta in frame_meta.objects:
                if obj_meta.is_primary:
                    continue

                mask_attr = obj_meta.get_attr_meta('yolov8_seg', 'mask')
                if not mask_attr:
                    continue

                bbox = obj_meta.bbox.as_ltrb_int()

                mask_overlay = np.where(
                    mask_attr.value[..., None], self.mask_color, self.bg_color
                )[0 : bbox[3] - bbox[1], 0 : bbox[2] - bbox[0]]

                alpha_comp(
                    frame_mat,
                    overlay=mask_overlay,
                    start=(bbox[0], bbox[1]),
                    stream=stream,
                )

                draw_rect(
                    frame_mat,
                    rect=bbox,
                    color=self.bbox_color,
                    thickness=2,
                    stream=stream,
                )
