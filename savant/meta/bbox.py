"""Bounding box metadata."""
from dataclasses import dataclass


@dataclass
class BaseBBox:
    """The class for regular (aligned) bounding box."""

    x_center: float
    """x coordinate of bounding box center point."""

    y_center: float
    """y coordinate of bounding box center point."""

    width: float
    """width of bounding box."""

    height: float
    """height of bounding box."""


class BBox(BaseBBox):
    """The class of regular (aligned) bounding box.

    Top and left properties were added to simplify interaction with
    BBox.
    """

    @property
    def top(self) -> float:
        """y coordinate of the upper left corner."""
        return self.y_center - 0.5 * self.height

    @top.setter
    def top(self, value: float):
        """Set y coordinate of center point using y coordinate of the upper
        left corner."""
        self.y_center = value + 0.5 * self.height

    @property
    def left(self) -> float:
        """x coordinate of the upper left corner."""
        return self.x_center - 0.5 * self.width

    @left.setter
    def left(self, value: float):
        """Set x coordinate of center point using x coordinate of the upper
        left corner."""
        self.x_center = value + 0.5 * self.width


@dataclass
class RBBox(BaseBBox):
    """The class of rotated bounding box."""

    angle: float
    """Rotation angle of bounding box around center point in degree."""
