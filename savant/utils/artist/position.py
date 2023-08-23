"""Anchor point position enum and utility."""
from enum import Enum
from typing import Tuple


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


def get_bottom_left_point(
    anchor_point_type: Position,
    anchor: Tuple[int, int],
    box_size: Tuple[int, int],
    baseline: int = 0,
) -> Tuple[int, int]:
    """Calculate text origin coordinates.

    :param anchor_point_type: Anchor point type of a rectangle with text.
    :param anchor: Anchor point X,Y coordinates.
    :param box_size: Box width and height.
    :param baseline: y-coordinate of the baseline relative to the bottom-most box point.
        Used for text boxes.
    :return: Bottom-left corner of the box in the image.
    """
    box_w, box_h = box_size
    anchor_x, anchor_y = anchor
    if anchor_point_type == Position.CENTER:
        left, bottom = anchor_x - box_w / 2, anchor_y + box_h / 2
    if anchor_point_type == Position.LEFT_TOP:
        left, bottom = anchor_x, anchor_y + box_h
    if anchor_point_type == Position.CENTER_TOP:
        left, bottom = anchor_x - box_w / 2, anchor_y + box_h
    if anchor_point_type == Position.RIGHT_TOP:
        left, bottom = anchor_x - box_w, anchor_y + box_h - baseline / 2
    if anchor_point_type == Position.LEFT_CENTER:
        left, bottom = anchor_x, anchor_y + box_h / 2 - baseline / 2
    if anchor_point_type == Position.RIGHT_CENTER:
        left, bottom = anchor_x - box_w, anchor_y + box_h / 2 - baseline / 2
    if anchor_point_type == Position.LEFT_BOTTOM:
        left, bottom = anchor_x, anchor_y - baseline
    if anchor_point_type == Position.CENTER_BOTTOM:
        left, bottom = anchor_x - box_w / 2, anchor_y - baseline
    if anchor_point_type == Position.RIGHT_BOTTOM:
        left, bottom = anchor_x - box_w, anchor_y - baseline

    return int(left), int(bottom)
