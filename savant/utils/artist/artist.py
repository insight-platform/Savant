from typing import Tuple, Optional, Union, List
from abc import ABC, abstractmethod
from savant.meta.bbox import BBox, RBBox
from .utils import Position


class Artist(ABC):
    @abstractmethod
    def add_text(
        self,
        text: str,
        anchor_x: int,
        anchor_y: int,
        size: int = 13,
        text_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        border_width: int = 0,
        border_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        bg_color: Optional[Tuple[float, float, float]] = None,
        padding: int = 3,
        anchor_point: Position = Position.CENTER,
    ):
        """Add text on the frame.

        :param text: display text.
        :param anchor_x: x coordinate of text position.
        :param anchor_y: y coordinate of text position.
        :param size: font size.
        :param text_color: font color.
        :param border_width: border width around the text.
        :param border_color: border color around the text.
        :param bg_color: background color.
        :param padding: increase the size of the rectangle around
            the text in each direction, in pixels.
        :param anchor_point: Anchor point of a  rectangle with text.
            For example, if you select Position.CENTER, the rectangle with the text
            will be drawn so that the center of the rectangle is at (x,y).
        """

    # pylint:disable=too-many-arguments
    @abstractmethod
    def add_bbox(
        self,
        bbox: Union[BBox, RBBox],
        border_width: int = 3,
        border_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),  # BGR, Green
        bg_color: Optional[Tuple[float, float, float]] = None,  # BGR
        padding: int = 3,
    ):
        """Draw bbox on frame.

        :param bbox: bounding box.
        :param border_width:  border width.
        :param border_color:  border color.
        :param bg_color: background color. If None, the rectangle will be transparent.
        :param padding: increase the size of the rectangle in each direction,
            value in pixels.
        """

    @abstractmethod
    def add_polygon(
        self,
        vertices: List[Tuple[float, float]],
        line_width: int = 3,
        line_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),  # BGR, Red
        bg_color: Optional[Tuple[float, float, float]] = None,  # BGR
    ):
        """Draw polygon.

        :param vertices: List of points
        :param line_width: line width
        :param line_color: line color
        :param bg_color: background color
        """

    @abstractmethod
    def blur(self, bbox: BBox, padding: int = 0):
        """Apply gaussian blur to the specified ROI."""
