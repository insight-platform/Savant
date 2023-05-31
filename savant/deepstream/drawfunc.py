"""Default implementation PyFunc for drawing on frame."""
from typing import Any, Dict, Optional, Tuple
import re
import pyds
import cv2
from savant_rs.primitives import (
    BoundingBoxDraw,
    ColorDraw,
    LabelDraw,
    DotDraw,
    PaddingDraw,
    ObjectDraw,
)

from savant.meta.object import ObjectMeta
from savant.deepstream.base_drawfunc import BaseNvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.meta.bbox import BBox
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.utils.artist import Position, Artist
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
        self.draw_spec = {}
        if self.rendered_objects:
            for unit, objects in self.rendered_objects.items():
                for obj, obj_draw_spec_cfg in objects.items():
                    self.draw_spec[(unit, obj)] = get_obj_draw_spec(obj_draw_spec_cfg)
        else:
            self.default_spec_track_id = ObjectDraw(
                bounding_box=BoundingBoxDraw(
                    color=ColorDraw(red=0, blue=0, green=255, alpha=255),
                    thickness=2,
                ),
                label=LabelDraw(
                    color=ColorDraw(red=255, blue=255, green=255, alpha=255),
                    font_scale=0.5,
                    thickness=1,
                    format=['{label} #{track_id}'],
                ),
            )
            self.default_spec_no_track_id = clone_obj_draw_spec(
                self.default_spec_track_id
            )
            self.default_spec_no_track_id.label.format = ['{label}']

        self.frame_streams = []

    def __call__(self, nvds_frame_meta: pyds.NvDsFrameMeta, buffer: Gst.Buffer):
        with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
            stream = cv2.cuda.Stream()
            self.frame_streams.append(stream)
            with Artist(frame_mat, stream) as artist:
                self.draw_on_frame(NvDsFrameMeta(nvds_frame_meta), artist)

    def finalize(self):
        """Finalize batch processing. Wait for all frame CUDA streams to finish."""
        for stream in self.frame_streams:
            stream.waitForCompletion()
        self.frame_streams = []

    def override_draw_spec(
        self, object_meta: ObjectMeta, specification: ObjectDraw
    ) -> ObjectDraw:
        """Override draw specification for an object
        based on dynamically changning object properties.
        For example, re-assign bbox color from default per object class one
        to custom per track id one.

        :param object_meta: Object's meta
        :param specification: Draw specification
        :return: Overridden draw specification
        """
        return specification

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws bounding boxes and labels for all objects in the frame's metadata.

        :param frame_meta: Frame metadata.
        :param artist: Artist to draw on the frame.
        """
        for obj_meta in frame_meta.objects:

            if obj_meta.is_primary:
                continue

            if len(self.draw_spec) > 0:
                spec = self.draw_spec.get((obj_meta.element_name, obj_meta.label), None)
            elif obj_meta.track_id != UNTRACKED_OBJECT_ID:
                spec = self.default_spec_track_id
            else:
                spec = self.default_spec_no_track_id

            if spec is None:
                continue

            spec = self.override_draw_spec(obj_meta, clone_obj_draw_spec(spec))

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
            bbox=obj_meta.bbox,
            border_color=rgba_color(spec.color),
            border_width=spec.thickness,
            padding=padding,
        )

    def _draw_label(self, obj_meta: ObjectMeta, artist: Artist, spec: LabelDraw):
        if isinstance(obj_meta.bbox, BBox):
            anchor_x = int(obj_meta.bbox.left)
            anchor_y = int(obj_meta.bbox.top)
            anchor_point = Position.LEFT_TOP
        else:
            anchor_x = int(obj_meta.bbox.x_center)
            anchor_y = int(obj_meta.bbox.y_center)
            anchor_point = Position.CENTER

        for format_str in spec.format:
            text = format_str.format(
                model=obj_meta.element_name,
                label=obj_meta.label,
                confidence=obj_meta.confidence,
                track_id=obj_meta.track_id,
            )

            text_bottom = artist.add_text(
                text=text,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                anchor_point=anchor_point,
                font_color=rgba_color(spec.color),
                font_scale=spec.font_scale,
                font_thickness=spec.thickness,
            )

            anchor_y = text_bottom

    def _draw_central_dot(self, obj_meta: ObjectMeta, artist: Artist, spec: DotDraw):
        artist.add_circle(
            (round(obj_meta.bbox.x_center), round(obj_meta.bbox.y_center)),
            spec.radius,
            rgba_color(spec.color),
            cv2.FILLED,
        )

    def _blur(self, obj_meta: ObjectMeta, artist: Artist):
        artist.blur(obj_meta.bbox)


def convert_hex_to_rgba(hex_color: str) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA.
    Hex color string is expected to be exactly 8 characters long.

    :param hex_color: Hex color string
    :return: RGBA color tuple
    """
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4, 6))


def get_obj_draw_spec(config: dict) -> ObjectDraw:
    bbox_draw = None

    if 'bbox' in config:
        padding_draw = None
        if 'padding' in config['bbox']:
            padding_draw = PaddingDraw(**config['bbox']['padding'])

        bbox_draw = BoundingBoxDraw(
            color=ColorDraw(*convert_hex_to_rgba(config['bbox']['color'])),
            padding=padding_draw,
            thickness=config['bbox']['thickness'],
        )

    central_dot_draw = None
    if 'central_dot' in config:
        central_dot_draw = DotDraw(
            color=ColorDraw(*convert_hex_to_rgba(config['central_dot']['color'])),
            radius=config['central_dot']['radius'],
        )

    label_draw = None
    if 'label' in config:
        label_draw = LabelDraw(
            color=ColorDraw(*convert_hex_to_rgba(config['label']['color'])),
            font_scale=config['label']['font_scale'],
            thickness=config['label']['thickness'],
            format=config['label']['format'],
        )

    blur = config.get('blur', False)

    return ObjectDraw(
        bounding_box=bbox_draw,
        label=label_draw,
        central_dot=central_dot_draw,
        blur=blur,
    )


def rgba_color(color_spec: ColorDraw) -> Tuple[int, int, int, int]:
    return (color_spec.red, color_spec.green, color_spec.blue, color_spec.alpha)


def clone_obj_draw_spec(spec: ObjectDraw) -> ObjectDraw:
    return ObjectDraw(
        bounding_box=clone_bbox_draw(spec.bounding_box),
        label=clone_label_draw(spec.label),
        central_dot=clone_dot_draw(spec.central_dot),
        blur=spec.blur,
    )


def clone_bbox_draw(spec: Optional[BoundingBoxDraw]) -> Optional[BoundingBoxDraw]:
    if spec is None:
        return None
    return BoundingBoxDraw(
        color=clone_color_draw(spec.color),
        padding=clone_padding_draw(spec.padding),
        thickness=spec.thickness,
    )


def clone_label_draw(spec: Optional[LabelDraw]) -> Optional[LabelDraw]:
    if spec is None:
        return None
    return LabelDraw(
        color=clone_color_draw(spec.color),
        font_scale=spec.font_scale,
        thickness=spec.thickness,
        format=spec.format.copy(),
    )


def clone_dot_draw(spec: Optional[DotDraw]) -> Optional[DotDraw]:
    if spec is None:
        return None
    return DotDraw(
        color=clone_color_draw(spec.color),
        radius=spec.radius,
    )


def clone_color_draw(spec: ColorDraw) -> ColorDraw:
    return ColorDraw(
        red=spec.red,
        green=spec.green,
        blue=spec.blue,
        alpha=spec.alpha,
    )


def clone_padding_draw(spec: Optional[PaddingDraw]) -> Optional[PaddingDraw]:
    if spec is None:
        return None
    return PaddingDraw(
        left=spec.left,
        top=spec.top,
        right=spec.right,
        bottom=spec.bottom,
    )
