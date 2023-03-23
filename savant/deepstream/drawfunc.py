"""Default implementation PyFunc for drawing on frame."""
from typing import Any, Dict, Optional, Tuple
import pyds
from savant.meta.object import ObjectMeta
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
        with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
            with Artist(frame_mat) as artist:
                self.draw_on_frame(NvDsFrameMeta(nvds_frame_meta), artist)

    def get_bbox_border_color(
        self, obj_meta: ObjectMeta
    ) -> Optional[Tuple[float, float, float]]:
        """Get object's bbox color.
        Draw only objects in rendered_objects if set.

        :param obj_meta: Object's meta
        :return: None, if there is no need to draw the object, otherwise color in BGR
        """
        if self.rendered_objects is None:
            return 0.0, 1.0, 0.0  # BGR
        # draw only rendered_objects if set
        if (
            obj_meta.element_name in self.rendered_objects
            and obj_meta.label in self.rendered_objects[obj_meta.element_name]
        ):
            return self.rendered_objects[obj_meta.element_name][obj_meta.label]

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws bounding boxes and labels for all objects in the frame's metadata.

        :param frame_meta: Frame metadata.
        :param artist: Artist to draw on the frame.
        """
        for obj_meta in frame_meta.objects:
            if obj_meta.is_primary:
                continue

            bbox_border_color = self.get_bbox_border_color(obj_meta)
            if bbox_border_color:
                artist.add_bbox(
                    bbox=obj_meta.bbox,
                    border_color=bbox_border_color,
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
