"""Bounding box metadata."""
import math
from typing import Tuple, Union
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

    @property
    def bottom(self) -> float:
        """y coordinate of the bbox bottom edge."""
        return self.y_center + 0.5 * self.height

    @property
    def right(self) -> float:
        """x coordinate of the bbox right edge."""
        return self.x_center + 0.5 * self.width

    def scale(self, scale_x: float, scale_y: float):
        """Scales BBox.

        :param scale_x: The scaling factor applied along the x-axis.
        :param scale_y: The scaling factor applied along the y-axis.
        """
        self.x_center *= scale_x
        self.y_center *= scale_y
        self.width *= scale_x
        self.height *= scale_y

    @property
    def tlbr(self) -> Tuple[float, float, float, float]:
        """[x_min, y_min, x_max, y_max] representation of the bbox."""
        return self.left, self.top, self.right, self.bottom

    def polygon(self) -> np.ndarray:
        """Representation of the bbox using 4 corner points.

        :return: polygon np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]), value in float
        """
        left, top, right, bottom = self.tlbr
        return np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ]
        )


@dataclass
class RBBox(BaseBBox):
    """The class of rotated bounding box."""

    angle: float
    """Rotation angle of bounding box around center point in degree."""

    def scale(self, scale_x: float, scale_y: float):
        """Scales BBox.

        :param scale_x: The scaling factor applied along the x-axis.
        :param scale_y: The scaling factor applied along the y-axis.
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

    def polygon(self) -> np.ndarray:
        """representation of the bbox using 4 corner points.
        :return: polygon np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        """
        angle_rad = self.angle / 180 * np.pi
        c, s = math.cos(angle_rad ), math.sin(angle_rad )
        rotation_matrix = np.array([[c, -s], [s, c]])
        pts = np.array(
            [
                [-self.width / 2, -self.height / 2],
                [self.width / 2, -self.height / 2],
                [self.width / 2, self.height / 2],
                [-self.width / 2, self.height / 2],
            ]
        )
        c_point = np.array([self.x_center, self.y_center])
        res_matmult = np.matmul(pts, rotation_matrix)
        rect_points = c_point + res_matmult
        return rect_points

    def to_bbox(self) -> BBox:
        """Converts rotated bounding box to regular (aligned) bounding box.
        :return: BBox
        """
        polygon = self.polygon()
        bbox_aligned_width, bbox_aligned_height = map(
            float, np.max(polygon, axis=0) - np.min(polygon, axis=0)
        )
        bbox_aligned_x_center, bbox_aligned_y_center = map(
            float, np.mean(polygon, axis=0)
        )
        return BBox(
            x_center=bbox_aligned_x_center,
            y_center=bbox_aligned_y_center,
            width=bbox_aligned_width,
            height=bbox_aligned_height,
        )
