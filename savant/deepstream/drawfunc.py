"""Default implementation PyFunc for drawing on frame."""
from typing import Any, Dict, Optional

import cv2
from savant_rs.draw_spec import (BoundingBoxDraw, DotDraw, LabelDraw,
                                 LabelPositionKind, ObjectDraw)
from savant_rs.primitives.geometry import RBBox

from savant.deepstream.base_drawfunc import BaseNvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.gstreamer import Gst  # noqa: F401
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.meta.object import ObjectMeta
from savant.utils.artist import Artist, Position
from savant.utils.draw_spec import get_default_draw_spec, get_obj_draw_spec


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
        self.draw_spec = {}
        if self.rendered_objects:
            for unit, objects in self.rendered_objects.items():
                for obj, obj_draw_spec_cfg in objects.items():
                    self.draw_spec[(unit, obj)] = get_obj_draw_spec(obj_draw_spec_cfg)
        else:
            self.default_spec_track_id = get_default_draw_spec(track_id=True)
            self.default_spec_no_track_id = get_default_draw_spec(track_id=False)

    def draw(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            stream = self.get_cuda_stream(frame_meta)
            with Artist(frame_mat, stream) as artist:
                self.draw_on_frame(frame_meta, artist)

    def override_draw_spec(
        self, object_meta: ObjectMeta, draw_spec: ObjectDraw
    ) -> ObjectDraw:
        """Override draw specification for an object
        based on dynamically changning object properties.
        For example, re-assign bbox color from default per object class one
        to custom per track id one.

        :param object_meta: Object's meta
        :param specification: Draw specification
        :return: Overridden draw specification
        """
        return draw_spec

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws bounding boxes and labels for all objects in the frame's metadata.

        :param frame_meta: Frame metadata.
        :param artist: Artist to draw on the frame.
        """
        for obj_meta in frame_meta.objects:

            if obj_meta.is_primary:
                continue

            if len(self.draw_spec) > 0:
                if (obj_meta.element_name, obj_meta.draw_label) in self.draw_spec:
                    spec = self.draw_spec[(obj_meta.element_name, obj_meta.draw_label)]
                    spec = self.override_draw_spec(obj_meta, spec.copy())
                else:
                    continue

            elif obj_meta.track_id != UNTRACKED_OBJECT_ID:
                spec = self.default_spec_track_id
            else:
                spec = self.default_spec_no_track_id

            # draw according to the specification
            # blur should be the first to be applied
            # to avoid blurring the other elements
            if spec.blur:
                self._blur(obj_meta, artist)
            if spec.bounding_box:
                self._draw_bounding_box(obj_meta, artist, spec.bounding_box)
            if spec.label:
                self._draw_label(obj_meta, artist, spec.label)
            if spec.central_dot:
                self._draw_central_dot(obj_meta, artist, spec.central_dot)

    def _draw_bounding_box(
        self, obj_meta: ObjectMeta, artist: Artist, spec: BoundingBoxDraw
    ):
        if spec.padding is not None:
            padding = spec.padding.padding
        else:
            padding = (0, 0, 0, 0)
        artist.add_bbox(
            obj_meta.bbox,
            spec.thickness,
            spec.border_color.rgba,
            spec.background_color.rgba,
            padding,
        )

    def _draw_label(self, obj_meta: ObjectMeta, artist: Artist, spec: LabelDraw):
        if (
            isinstance(obj_meta.bbox, RBBox)
            or spec.position.position == LabelPositionKind.Center
        ):
            anchor_x = int(obj_meta.bbox.xc)
            anchor_y = int(obj_meta.bbox.yc)
            anchor_point = Position.CENTER
        else:
            anchor_x = int(obj_meta.bbox.left)
            anchor_y = int(obj_meta.bbox.top)
            if spec.position.position == LabelPositionKind.TopLeftInside:
                anchor_point = Position.LEFT_TOP
            else:
                # consider default position as TopLeftOutside
                anchor_point = Position.LEFT_BOTTOM

        anchor_x += spec.position.margin_x
        anchor_y += spec.position.margin_y

        if spec.padding is not None:
            padding = spec.padding.padding
        else:
            padding = (0, 0, 0, 0)

        if spec.position.position == LabelPositionKind.TopLeftOutside:
            lines_sequence = reversed(spec.format)
            offset_sign = -1
        else:
            lines_sequence = spec.format
            offset_sign = 1

        for format_str in lines_sequence:
            text = format_str.format(
                model=obj_meta.element_name,
                label=obj_meta.draw_label,
                confidence=obj_meta.confidence,
                track_id=obj_meta.track_id,
            )
            text_box_height = artist.add_text(
                text,
                (anchor_x, anchor_y),
                spec.font_scale,
                spec.thickness,
                spec.font_color.rgba,
                1,
                spec.border_color.rgba,
                spec.background_color.rgba,
                padding,
                anchor_point,
            )
            anchor_y += offset_sign * text_box_height

    def _draw_central_dot(self, obj_meta: ObjectMeta, artist: Artist, spec: DotDraw):
        artist.add_circle(
            (round(obj_meta.bbox.xc), round(obj_meta.bbox.yc)),
            spec.radius,
            spec.color.rgba,
            cv2.FILLED,
        )

    def _blur(self, obj_meta: ObjectMeta, artist: Artist):
        artist.blur(obj_meta.bbox)
