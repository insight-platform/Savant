"""Default implementation PyFunc for drawing on frame."""
from typing import Any, Dict, Optional, Tuple
import copy
import pyds
import cv2
from savant_rs.primitives import (
    BoundingBoxDraw,
    ColorDraw,
    LabelDraw,
    DotDraw,
    PaddingDraw,
    ObjectDraw,
    LabelPosition,
    LabelPositionKind
)

from savant.meta.object import ObjectMeta
from savant.deepstream.base_drawfunc import BaseNvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.meta.bbox import RBBox
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
                    print(obj_draw_spec_cfg)
                    print(self.draw_spec[(unit, obj)])
                    print(self.draw_spec[(unit, obj)].bounding_box.background_color.rgba)

        else:
            default_bbox_spec = BoundingBoxDraw(
                color=ColorDraw(red=0, blue=0, green=255, alpha=255),
                thickness=2,
            )
            default_text_color = ColorDraw(red=255, blue=255, green=255, alpha=255)
            default_font_scale = 0.5
            default_thickness = 1
            self.default_spec_track_id = ObjectDraw(
                bounding_box=default_bbox_spec,
                label=LabelDraw(
                    color=default_text_color,
                    font_scale=default_font_scale,
                    thickness=default_thickness,
                    format=['{label} #{track_id}'],
                ),
            )
            self.default_spec_no_track_id = ObjectDraw(
                bounding_box=default_bbox_spec,
                label=LabelDraw(
                    color=default_text_color,
                    font_scale=default_font_scale,
                    thickness=default_thickness,
                    format=['{label}'],
                ),
            )

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
                if (obj_meta.element_name, obj_meta.label) in self.draw_spec:
                    spec = self.draw_spec[(obj_meta.element_name, obj_meta.label)]
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
            bbox=obj_meta.bbox,
            border_color=spec.border_color.rgba,
            border_width=spec.thickness,
            bg_color=spec.background_color.rgba,
            padding=padding,
        )

    def _draw_label(self, obj_meta: ObjectMeta, artist: Artist, spec: LabelDraw):
        if isinstance(obj_meta.bbox, RBBox) or spec.position == LabelPositionKind.Center:
            anchor_x = int(obj_meta.bbox.x_center)
            anchor_y = int(obj_meta.bbox.y_center)
            anchor_point = Position.CENTER
        else:
            if spec.position == LabelPositionKind.TopLeftOutside:
                anchor_x = int(obj_meta.bbox.left) - spec.position.margin_x
                anchor_y = int(obj_meta.bbox.top) - spec.position.margin_y
                anchor_point=Position.LEFT_BOTTOM
            else:
                # consider default position as TopLeftInside
                anchor_point = Position.LEFT_TOP
                anchor_x = int(obj_meta.bbox.left) + spec.position.margin_x
                anchor_y = int(obj_meta.bbox.top) + spec.position.margin_y

        if spec.padding is not None:
            padding = spec.padding.padding
        else:
            padding = (0, 0, 0, 0)

        for format_str in spec.format:
            text = format_str.format(
                model=obj_meta.element_name,
                label=obj_meta.draw_label,
                confidence=obj_meta.confidence,
                track_id=obj_meta.track_id,
            )

            text_bottom = artist.add_text(
                text=text,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                anchor_point=anchor_point,
                font_color=spec.font_color.rgba,
                font_scale=spec.font_scale,
                font_thickness=spec.thickness,
                border_color=spec.border_color.rgba,
                bg_color=spec.background_color.rgba,
                padding=padding,
                border_width=1,
            )

            anchor_y = text_bottom

    def _draw_central_dot(self, obj_meta: ObjectMeta, artist: Artist, spec: DotDraw):
        artist.add_circle(
            (round(obj_meta.bbox.x_center), round(obj_meta.bbox.y_center)),
            spec.radius,
            spec.color.rgba,
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
        # default green borders
        border_color = config['bbox'].get('border_color', '00FF00FF')
        # default transparent background
        background_color = config['bbox'].get('background_color', '00000000')
        # default border thickness 2
        thickness = config['bbox'].get('thickness', 2)
        # default no padding

        if 'padding' in config['bbox']:
            padding_draw = PaddingDraw(**config['bbox']['padding'])
        else:
            padding_draw = PaddingDraw()
        bbox_draw = BoundingBoxDraw(
            border_color=ColorDraw(*convert_hex_to_rgba(border_color)),
            background_color=ColorDraw(*convert_hex_to_rgba(background_color)),
            padding=padding_draw,
            thickness=thickness,
        )

    central_dot_draw = None
    if 'central_dot' in config:
        # default green color
        color = config['central_dot'].get('color', '00FF00FF')
        # default radius 5
        radius = config['central_dot'].get('radius', 5)
        central_dot_draw = DotDraw(
            color=ColorDraw(*convert_hex_to_rgba(color)),
            radius=radius,
        )

    label_draw = None
    if 'label' in config:
        # default white font color
        font_color = config['label'].get('font_color', 'FFFFFFFF')
        # default transparent border
        border_color = config['label'].get('border_color', '00000000')
        # default black background
        background_color = config['label'].get('background_color', '000000FF')
        # default font scale 0.5
        font_scale = config['label'].get('font_scale', 0.5)
        # default font thickness 1
        thickness = config['label'].get('thickness', 1)
        # default format: {label}
        label_format = config['label'].get('format', ['{label}'])

        # rely on rust for defaults for label position
        if 'position' in config['label']:
            label_pos_kwargs = copy.deepcopy(config['label']['position'])
            if 'position' in label_pos_kwargs:
                if label_pos_kwargs['position'] == 'Center':
                    label_pos_kwargs['position'] = LabelPositionKind.Center
                elif label_pos_kwargs['position'] == 'TopLeftOutside':
                    label_pos_kwargs['position'] = LabelPositionKind.TopLeftOutside
                elif label_pos_kwargs['position'] == 'TopLeftInside':
                    label_pos_kwargs['position'] = LabelPositionKind.TopLeftInside
                else:
                    # invalid position kind
                    label_pos_kwargs.pop('position')
            label_position = LabelPosition(**label_pos_kwargs)
        else:
            label_position = LabelPosition()

        label_draw = LabelDraw(
            font_color=ColorDraw(*convert_hex_to_rgba(font_color)),
            border_color=ColorDraw(*convert_hex_to_rgba(border_color)),
            background_color=ColorDraw(*convert_hex_to_rgba(background_color)),
            font_scale=font_scale,
            thickness=thickness,
            format=label_format,
            position=label_position,
        )

    blur = config.get('blur', False)

    return ObjectDraw(
        bounding_box=bbox_draw,
        label=label_draw,
        central_dot=central_dot_draw,
        blur=blur,
    )
