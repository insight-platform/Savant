"""Cairo-dependent Artist implementation."""
import math
from enum import Enum
from typing import Tuple, Optional, Union, List

from cairo import Context

from savant.meta.bbox import BBox, RBBox


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

# BGR format
COLOR = {
    'red': (0, 0, 255),
    'green': (0, 128, 0),
    'blue': (255, 0, 0),
    'darkred': (0, 0, 139),
    'orangered': (0, 69, 255),
    'orange': (0, 165, 255),
    'yellow': (0, 255, 255),
    'lime': (0, 255, 0),
    'magenta': (255, 0, 255)
}


class Artist:
    """Drawer on frame cairo context using primitives.

    :param context: Cairo frame context
    """

    def __init__(self, context: Context):
        self.context = context

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

        self.context.set_font_size(size)
        (text_x, text_y, width, height, _, _) = self.context.text_extents(text)
        if anchor_point == Position.CENTER:
            text_x, text_y = anchor_x - width / 2, anchor_y + height / 2
        elif anchor_point == Position.LEFT_TOP:
            text_x, text_y = anchor_x, anchor_y + height
        elif anchor_point == Position.CENTER_TOP:
            text_x, text_y = anchor_x - width / 2, anchor_y + height
        elif anchor_point == Position.RIGHT_TOP:
            text_x, text_y = anchor_x - width, anchor_y + height
        elif anchor_point == Position.LEFT_CENTER:
            text_x, text_y = anchor_x, anchor_y + height / 2
        elif anchor_point == Position.RIGHT_CENTER:
            text_x, text_y = anchor_x - width, anchor_y + height / 2
        elif anchor_point == Position.LEFT_BOTTOM:
            text_x, text_y = anchor_x, anchor_y
        elif anchor_point == Position.CENTER_BOTTOM:
            text_x, text_y = anchor_x - width / 2, anchor_y
        elif anchor_point == Position.RIGHT_BOTTOM:
            text_x, text_y = anchor_x - width, anchor_y
        text_x -= padding
        text_y -= padding
        if bg_color or border_width:
            self.add_bbox(
                bbox=BBox(
                    x_center=text_x + width / 2,
                    y_center=text_y - height / 2,
                    width=width,
                    height=height,
                ),
                border_width=border_width,
                border_color=border_color,
                bg_color=bg_color,
                padding=padding,
            )
        self.context.set_source_rgb(*text_color)
        self.context.move_to(text_x, text_y)
        self.context.show_text(text)
        self.context.stroke()

    # pylint:disable=too-many-arguments
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
        if isinstance(bbox, BBox):
            if border_width:
                self.context.set_line_width(border_width)
                self.context.set_source_rgb(*border_color)
                self.context.rectangle(
                    bbox.left - padding - border_width,
                    bbox.top - padding - border_width,
                    bbox.width + 2 * (padding + border_width),
                    bbox.height + 2 * (padding + border_width),
                )
                self.context.stroke()
            if bg_color:
                self.context.rectangle(
                    bbox.left - padding,
                    bbox.top - padding,
                    bbox.width + 2 * padding,
                    bbox.height + 2 * padding,
                )
                self.context.set_source_rgb(*bg_color)
                self.context.fill()
                self.context.stroke()
        elif isinstance(bbox, RBBox):
            pad_bbox = bbox
            if padding:
                pad_bbox = RBBox(
                    x_center=bbox.x_center,
                    y_center=bbox.y_center,
                    width=bbox.width + 2 * padding,
                    height=bbox.height + 2 * padding,
                    angle=pad_bbox.angle,
                )
            vertices = bbox_to_vertices(pad_bbox)
            self.add_polygon(
                vertices=vertices,
                line_width=border_width,
                line_color=border_color,
                bg_color=bg_color,
            )

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
        self.context.set_line_width(line_width)
        self.context.set_source_rgb(*line_color)
        if vertices:
            self.context.move_to(*vertices[0])
        for vert_x, vert_y in vertices[1:]:
            self.context.line_to(int(vert_x), int(vert_y))
        self.context.close_path()
        self.context.stroke()
        if bg_color:
            for vert_x, vert_y in vertices[1:]:
                self.context.line_to(vert_x, vert_y)
            self.context.close_path()
            self.context.fill()
            self.context.stroke()


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
