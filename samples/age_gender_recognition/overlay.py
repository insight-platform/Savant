"""Overlay with age/gender labels and optional YOLOv8face landmarks rendering."""

from typing import List, Sequence, Tuple, Union

import cv2
from savant_rs.draw_spec import LabelDraw, ObjectDraw

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.meta.object import ObjectMeta
from savant.parameter_storage import param_storage

# Detector name from module.yml (parameters.detection_model_name)
MODEL_NAME = param_storage()['detection_model_name']  # e.g., 'yolov8nface'


def _parse_rgba(color: Union[str, Sequence[int]]) -> tuple:
    """Accepts 'RRGGBBAA' or a sequence [R, G, B, A] and returns (R, G, B, A)."""
    if isinstance(color, str):
        s = color.strip().lstrip('#')
        if len(s) != 8:
            raise ValueError("landmarks_color must be 8-hex 'RRGGBBAA'")
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        a = int(s[6:8], 16)
        return (r, g, b, a)
    return tuple(int(v) for v in color)  # type: ignore


def _extract_landmarks(obj: ObjectMeta) -> List[Tuple[float, float]]:
    """
    Returns a list of 5 (x, y) landmark points.
    Tries common namespaces where 'landmarks' may be stored.
    """
    lm_attr = (
        obj.get_attr_meta(MODEL_NAME, 'landmarks')  # detector namespace (typical)
        or obj.get_attr_meta('value', 'landmarks')  # generic namespace (fallback)
        or obj.get_attr_meta(
            'smoothed_value', 'landmarks'
        )  # if someone smooths landmarks later
    )
    if lm_attr is None:
        return []

    vals = lm_attr.value
    try:
        arr = list(vals)
    except Exception:
        return []

    # Flattened [x0, y0, x1, y1, ..., x4, y4]
    if len(arr) == 10:
        return [(float(arr[i]), float(arr[i + 1])) for i in range(0, 10, 2)]
    # Nested [[x, y], ...] x5
    if len(arr) == 5 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in arr):
        return [(float(p[0]), float(p[1])) for p in arr]

    return []


class Overlay(NvDsDrawFunc):
    def __init__(
        self,
        draw_landmarks: bool = False,
        landmarks_radius: int = 3,
        landmarks_color: Union[str, Sequence[int]] = 'FF0000FF',  # RGBA red
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.draw_landmarks = bool(draw_landmarks)
        self.landmarks_radius = int(landmarks_radius)
        self.landmarks_color = _parse_rgba(landmarks_color)

    def override_draw_spec(
        self, object_meta: ObjectMeta, draw_spec: ObjectDraw
    ) -> ObjectDraw:
        """Override draw spec for object with label 'Face'.
        Add age and gender attributes to the object visualisation.
        """
        if object_meta.label == 'face':
            new_label_format = draw_spec.label.format

            # one attribute per line
            # if there's no specific attribute for the object on this frame
            # reserve a line for it anyway
            # so that the object's labels don't jump up and down
            attr_meta = object_meta.get_attr_meta('smoothed_value', 'age')
            if attr_meta is not None:
                new_label_format += [f'age: {round(attr_meta.value)}']
            else:
                new_label_format += ['']

            attr_meta = object_meta.get_attr_meta('smoothed_value', 'gender')
            if attr_meta is not None:
                new_label_format += [f'gender: {str(attr_meta.value)}']
            else:
                new_label_format += ['']

            # draw_spec.label.format = new_label_format
            draw_spec = ObjectDraw(
                bounding_box=draw_spec.bounding_box,
                label=LabelDraw(
                    font_color=draw_spec.label.font_color,
                    border_color=draw_spec.label.border_color,
                    background_color=draw_spec.label.background_color,
                    padding=draw_spec.label.padding,
                    font_scale=draw_spec.label.font_scale,
                    thickness=draw_spec.label.thickness,
                    format=new_label_format,
                    position=draw_spec.label.position,
                ),
                central_dot=draw_spec.central_dot,
                blur=draw_spec.blur,
            )
        return draw_spec

    def draw_on_frame(self, frame_meta, artist):
        """Run default drawing first, then optionally draw face landmarks."""
        super().draw_on_frame(frame_meta, artist)

        if not self.draw_landmarks:
            return

        for obj in frame_meta.objects:
            if obj.element_name != MODEL_NAME or obj.label != 'face':
                continue

            pts = _extract_landmarks(obj)
            if not pts:
                continue

            for x, y in pts:
                artist.add_circle(
                    (int(round(x)), int(round(y))),
                    self.landmarks_radius,
                    self.landmarks_color,
                    cv2.FILLED,
                )
