"""Custom DrawFunc implementation."""
import cv2
import numpy as np
import pyds
from savant.gstreamer import Gst  # noqa: F401
from savant.deepstream.opencv_utils import nvds_to_gpu_mat, alpha_comp, draw_rect
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta


class Overlay(NvDsDrawFunc):
    """Custom implementation of PyFunc for drawing on frame."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bbox_color = (0, 255, 0, 255)
        self.mask_color = np.array([0, 255, 0, 64], dtype=np.uint8)
        self.bg_color = np.array([0, 0, 0, 0], dtype=np.uint8)

    def __call__(self, nvds_frame_meta: pyds.NvDsFrameMeta, buffer: Gst.Buffer):
        with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
            stream = cv2.cuda.Stream()
            self.frame_streams.append(stream)

            frame_meta = NvDsFrameMeta(nvds_frame_meta)
            for obj_meta in frame_meta.objects:
                if obj_meta.is_primary:
                    continue

                mask_attr = obj_meta.get_attr_meta('yolov8_seg', 'mask')
                if not mask_attr:
                    continue

                bbox = obj_meta.bbox.as_ltrb_int()

                mask_overlay = np.where(
                    mask_attr.value[..., None], self.mask_color, self.bg_color
                )

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
