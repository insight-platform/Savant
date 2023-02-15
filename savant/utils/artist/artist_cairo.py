"""Cairo-dependent Artist implementation."""
from typing import Tuple, Optional, Union, List
from contextlib import AbstractContextManager
import numpy as np
from cairo import Context, Format, ImageSurface
from savant.meta.bbox import BBox, RBBox
from .artist import Artist
from .utils import Position, bbox_to_vertices, get_text_origin


class ArtistCairo(Artist, AbstractContextManager):
    """Drawer on frame cairo context using primitives.

    :param frame: frame data as numpy array
    """

    def __init__(self, frame: np.ndarray):
        self.frame: np.ndarray = frame
        self.surface = None
        self.context = None

    def __enter__(self):
        frame_height, frame_width, frame_channels = self.frame.shape
        self.surface = ImageSurface.create_for_data(
            self.frame,
            Format.ARGB32,
            frame_width,
            frame_height,
            self.frame.strides[0],
        )
        self.context = Context(self.surface)
        return self

    def __exit__(self, *exc_details):
        self.surface.flush()
        self.surface.finish()

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
        (_, _, width, height, _, _) = self.context.text_extents(text)
        text_x, text_y = get_text_origin(anchor_point, anchor_x, anchor_y, width, height)
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

    def blur(self, bbox: BBox):
        """Not implemented"""
        raise NotImplementedError()
