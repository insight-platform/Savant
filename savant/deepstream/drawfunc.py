"""Default implementation PyFunc for drawing on frame."""
from typing import Any, Dict, Optional
import pyds
from savant.deepstream.base_drawfunc import BaseNvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.meta.bbox import BBox, RBBox
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.utils.artist import Position, Artist, COLOR
from savant.gstreamer import Gst  # noqa: F401
from savant.deepstream.opencv_utils import nvds_to_gpu_mat


class NvDsDrawFunc(BaseNvDsDrawFunc):
    """Default implementation of PyFunc for drawing on frame.
    Uses OpenCV GpuMat to work with frame data without mapping to CPU
    through OpenCV-based Artist.

    PyFunc implementations are defined in and instantiated by a
    :py:class:`.PyFunc` structure.
    """

    def __init__(self, **kwargs):
        self.rendered_objects: Optional[Dict[str, Dict[str, Any]]] = None
        super().__init__(**kwargs)
        if self.rendered_objects:
            for _, labels in self.rendered_objects.items():
                for label, color in labels.items():
                    labels[label] = COLOR[color]

    def __call__(self, nvds_frame_meta: pyds.NvDsFrameMeta, buffer: Gst.Buffer):
        frame_meta = NvDsFrameMeta(frame_meta=nvds_frame_meta)
        with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
            with Artist(frame_mat) as artist:
                self.draw_on_frame(frame_meta, artist)

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws bounding boxes and labels for all objects in the frame.

        :param frame_meta: Frame metadata for a frame in a batch.
        :param artist: Cairo context drawer to drawing primitives and directly on frame.
        """
        for obj_meta in frame_meta.objects:
            if not obj_meta.element_name and obj_meta.label == 'frame':
                continue

            if self.rendered_objects is None or (
                obj_meta.element_name in self.rendered_objects
                and obj_meta.label in self.rendered_objects[obj_meta.element_name]
            ):
                artist.add_bbox(
                    bbox=obj_meta.bbox,
                    border_color=self.rendered_objects[obj_meta.element_name][
                        obj_meta.label
                    ]
                    if self.rendered_objects
                    else (0.0, 1.0, 0.0),
                )

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
