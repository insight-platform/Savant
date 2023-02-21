"""Bounding box metadata."""
from dataclasses import dataclass
import numpy as np
from savant.converter.scale import scale_rbbox


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

    def scale(self, scale_x: float, scale_y: float):
        """Scales BBox.

        :param scale_x:
        :param scale_y:
        """
        self.left *= scale_x
        self.top *= scale_y
        self.width *= scale_x
        self.height *= scale_y


@dataclass
class RBBox(BaseBBox):
    """The class of rotated bounding box."""

    angle: float
    """Rotation angle of bounding box around center point in degree."""

    def scale(self, scale_x: float, scale_y: float):
        """Scales BBox.

        :param scale_x:
        :param scale_y:
        """
        scaled_bbox = scale_rbbox(
            bboxes=np.array(
                [
                    [
                        self.x_center,
                        self.y_center,
                        self.width,
                        self.height,
                        self.angle,
                    ]
                ]
            ),
            scale_factor_x=scale_x,
            scale_factor_y=scale_y,
        )[0]
        self.x_center = scaled_bbox[0]
        self.y_center = scaled_bbox[1]
        self.width = scaled_bbox[2]
        self.height = scaled_bbox[3]
        self.angle = scaled_bbox[4]
