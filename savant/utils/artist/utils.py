import math
from enum import Enum

from typing import Tuple, List

from savant.meta.bbox import RBBox


class Position(Enum):
    """Start position of the element drawing relative to the box."""

    CENTER = 1
    LEFT_TOP = 2
    CENTER_TOP = 3
    RIGHT_TOP = 4
    LEFT_CENTER = 5
    RIGHT_CENTER = 6
    LEFT_BOTTOM = 7
    CENTER_BOTTOM = 8
    RIGHT_BOTTOM = 9


def get_text_origin(anchor_point: Position, anchor_x, anchor_y, text_w, text_h):
    if anchor_point == Position.CENTER:
        return anchor_x - text_w / 2, anchor_y + text_h / 2
    if anchor_point == Position.LEFT_TOP:
        return anchor_x, anchor_y + text_h
    if anchor_point == Position.CENTER_TOP:
        return anchor_x - text_w / 2, anchor_y + text_h
    if anchor_point == Position.RIGHT_TOP:
        return anchor_x - text_w, anchor_y + text_h
    if anchor_point == Position.LEFT_CENTER:
        return anchor_x, anchor_y + text_h / 2
    if anchor_point == Position.RIGHT_CENTER:
        return anchor_x - text_w, anchor_y + text_h / 2
    if anchor_point == Position.LEFT_BOTTOM:
        return anchor_x, anchor_y
    if anchor_point == Position.CENTER_BOTTOM:
        return anchor_x - text_w / 2, anchor_y
    if anchor_point == Position.RIGHT_BOTTOM:
        return anchor_x - text_w, anchor_y


def bbox_to_vertices(rbbox: RBBox) -> List[Tuple[float, float]]:
    """Convert rotated bounding boxes to list of 2D points."""
    x_center, y_center, width, height, angle = (
        rbbox.x_center,
        rbbox.y_center,
        rbbox.width,
        rbbox.height,
        rbbox.angle / 180 * math.pi,
    )

    width_sin = width / 2 * math.sin(angle)
    width_cos = width / 2 * math.cos(angle)
    height_sin = height / 2 * math.sin(angle)
    height_cos = height / 2 * math.cos(angle)

    return [
        (x_center - width_cos + height_sin, y_center - width_sin - height_cos),
        (x_center + width_cos + height_sin, y_center + width_sin - height_cos),
        (x_center + width_cos - height_sin, y_center + width_sin + height_cos),
        (x_center - width_cos - height_sin, y_center - width_sin + height_cos),
    ]
