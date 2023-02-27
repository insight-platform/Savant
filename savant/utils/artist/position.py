"""Anchor point position enum and utility."""
from typing import Tuple
from enum import Enum


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


def get_text_origin(
    anchor_point: Position, anchor_x: int, anchor_y: int, text_w: int, text_h: int
) -> Tuple[int, int]:
    """Calculate text origin coordinates.

    :param anchor_point: Anchor point type of a rectangle with text.
    :param anchor_x: Anchor point X coordinate.
    :param anchor_y: Anchor point Y coordinate.
    :param text_w: Text box width.
    :param text_h: Text box height.
    :return: Text origin X,Y.
    """

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
