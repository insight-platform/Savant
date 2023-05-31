"""Default implementation PyFunc for drawing on frame."""
from typing import Any, Dict, Optional, Tuple
import pyds
import cv2
from savant_rs.primitives import BoundingBoxDraw, ColorDraw, LabelDraw, DotDraw, PaddingDraw, ObjectDraw

from savant.meta.object import ObjectMeta
from savant.deepstream.base_drawfunc import BaseNvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.meta.bbox import BBox
from savant.meta.constants import UNTRACKED_OBJECT_ID
from savant.utils.artist import Position, Artist
from savant.gstreamer import Gst  # noqa: F401
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
import pprint

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
            pprint.pprint(self.rendered_objects)
            for unit, objects in self.rendered_objects.items():
                for obj, obj_draw_spec_cfg in objects.items():
                    self.draw_spec[(unit, obj)] = self._get_obj_draw_spec(obj_draw_spec_cfg)

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

    def override_draw_spec(self, object_meta: ObjectMeta, specification: ObjectDraw) -> ObjectDraw:
        """Override draw specification for an object
        based on dynamically changning object properties.

        :param object_meta: Object's meta
        :param specification: Draw specification
        :return: Overridden draw specification
        """
        # make sure default draw spec is not modified
        return specification

    def draw_on_frame(self, frame_meta: NvDsFrameMeta, artist: Artist):
        """Draws bounding boxes and labels for all objects in the frame's metadata.

        :param frame_meta: Frame metadata.
        :param artist: Artist to draw on the frame.
        """
        for obj_meta in frame_meta.objects:

            if obj_meta.is_primary:
                continue

            spec = self.override_draw_spec(obj_meta, self.draw_spec[(obj_meta.element_name, obj_meta.label)])

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

    def _draw_bounding_box(self, obj_meta: ObjectMeta, artist: Artist, spec: BoundingBoxDraw):
        artist.add_bbox(
            bbox=obj_meta.bbox,
            border_color=rgba_color(spec.color),
            border_width=spec.thickness,
            padding=spec.padding,
        )

    def _draw_label(self, obj_meta: ObjectMeta, artist: Artist, spec: LabelDraw):
        if isinstance(obj_meta.bbox, BBox):
            anchor_x=int(obj_meta.bbox.left)
            anchor_y=int(obj_meta.bbox.top)
            anchor_point=Position.LEFT_TOP
        else:
            anchor_x=int(obj_meta.bbox.x_center)
            anchor_y=int(obj_meta.bbox.y_center)
            anchor_point=Position.CENTER

        for format_str in spec.format:
            # if the object is not tracked, the track_id is UNTRACKED_OBJECT_ID
            # do not add the track_id if it is UNTRACKED_OBJECT_ID
            text = format_str.format(
                model=obj_meta.element_name,
                label=obj_meta.label,
                confidence=obj_meta.confidence,
                track_id=obj_meta.track_id
            )

            text_size = artist.add_text(
                text=text,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                anchor_point=anchor_point,
                font_color=rgba_color(spec.color),
                font_scale=spec.font_scale,
                font_thickness=spec.thickness,
            )

            anchor_y += text_size[1]

    def _draw_central_dot(self, obj_meta: ObjectMeta, artist: Artist, spec: DotDraw):
        artist.add_circle(
            (round(obj_meta.bbox.x_center), round(obj_meta.bbox.y_center)),
            spec.radius,
            rgba_color(spec.color),
            cv2.FILLED
        )

    def _blur(self, obj_meta: ObjectMeta, artist: Artist):
        artist.blur(obj_meta.bbox)

def convert_hex_to_rgba(hex_color: str) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA.

    :param hex_color: Hex color string
    :return: RGBA color tuple
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4, 6))

def get_obj_draw_spec(config:dict) -> ObjectDraw:
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
            color = ColorDraw(*convert_hex_to_rgba(config['label']['color'])),
            font_scale=2.5,
            thickness=2,
            format=["{model}", "{label}", "{confidence}", "{track_id}"]
        )

    blur = config.get('blur', False)

    return ObjectDraw(bounding_box=bbox_draw, label=label_draw, central_dot=central_dot_draw, blur=blur)

def rgba_color(color_spec: ColorDraw):
    return (color_spec.red, color_spec.green, color_spec.blue, color_spec.alpha)
