from typing import Tuple, Optional, Union, List
from contextlib import AbstractContextManager
import numpy as np
import cv2
from savant.meta.bbox import BBox, RBBox
from .artist import Artist
from .utils import Position, get_text_origin


def convert_color(color: Tuple[float, float, float], alpha: int = 255):
    """Convert color from BGR floats to RGBA int8."""
    return int(color[2] * 255), int(color[1] * 255), int(color[0] * 255), alpha


class ArtistOpenCV(Artist, AbstractContextManager):
    def __init__(self, frame: cv2.cuda.GpuMat) -> None:
        self.stream = cv2.cuda.Stream()
        self.frame: cv2.cuda.GpuMat = frame
        self.width, self.height = self.frame.size()
        self.max_col = self.width - 1
        self.max_row = self.height - 1
        self.alpha_op = cv2.cuda.ALPHA_OVER_PREMUL
        self.overlay = None
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.gaussian_filter = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_details):
        # apply alpha comp if overlay is not null
        if self.overlay is not None:
            overlay = cv2.cuda.GpuMat(self.overlay)
            cv2.cuda.alphaComp(
                overlay, self.frame, self.alpha_op, self.frame, stream=self.stream
            )
        self.stream.waitForCompletion()

    def __init_overlay(self):
        self.overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def __init_gaussian(self):
        self.gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC4, cv2.CV_8UC4, (31, 31), 100, 100
        )

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
        if self.overlay is None:
            self.__init_overlay()

        font_scale = 0.4
        font_thickness = 1

        text_size, baseline = cv2.getTextSize(
            text, self.font_face, font_scale, font_thickness
        )

        text_x, text_y = get_text_origin(
            anchor_point, anchor_x, anchor_y, text_size[0], text_size[1]
        )
        text_x = int(text_x) - padding
        text_y = int(text_y) - padding

        if bg_color or border_width:
            rect_left = text_x - border_width
            rect_top = text_y + baseline
            rect_right = text_x - border_width + text_size[0]
            rect_bottom = text_y - text_size[1]
            rect_tl = rect_left, rect_top
            rect_br = rect_right, rect_bottom
            if bg_color is not None:
                cv2.rectangle(
                    self.overlay, rect_tl, rect_br, convert_color(bg_color), cv2.FILLED
                )
            if border_width > 0:
                cv2.rectangle(
                    self.overlay,
                    rect_tl,
                    rect_br,
                    convert_color(border_color),
                    border_width,
                )

        cv2.putText(
            self.overlay,
            text,
            (text_x, text_y),
            self.font_face,
            font_scale,
            convert_color(text_color),
            font_thickness,
        )

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
            left, top, right, bottom, _, _ = self.convert_bbox(
                bbox, padding, border_width
            )

            if bg_color is not None:
                self.frame.colRange(left, right).rowRange(top, bottom).setTo(
                    convert_color(bg_color), stream=self.stream
                )

            if border_color != bg_color:
                color = convert_color(border_color)
                self.frame.colRange(left, right).rowRange(
                    top, top + border_width
                ).setTo(color, stream=self.stream)
                self.frame.colRange(left, right).rowRange(
                    bottom - border_width, bottom
                ).setTo(color, stream=self.stream)
                self.frame.colRange(left, left + border_width).rowRange(
                    top, bottom
                ).setTo(color, stream=self.stream)
                self.frame.colRange(right - border_width, right).rowRange(
                    top, bottom
                ).setTo(color, stream=self.stream)

        elif isinstance(bbox, RBBox):
            x_center = bbox.x_center
            y_center = bbox.y_center
            width = bbox.width
            height = bbox.height
            degrees = bbox.angle
            if padding:
                width += 2 * padding
                height += 2 * padding

            vertices = cv2.boxPoints(((x_center, y_center), (width, height), degrees))

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
        if self.overlay is None:
            self.__init_overlay()
        vertices = np.intp(vertices)
        if bg_color is not None:
            cv2.drawContours(
                self.overlay, [vertices], 0, convert_color(bg_color), cv2.FILLED
            )
        cv2.drawContours(
            self.overlay, [vertices], 0, convert_color(line_color), line_width
        )

    def blur(self, bbox: BBox, padding: int = 0):
        """Apply gaussian blur to the specified ROI."""

        if self.gaussian_filter is None:
            self.__init_gaussian()

        left, top, _, _, width, height = self.convert_bbox(bbox, padding, 0)
        roi_mat = cv2.cuda.GpuMat(self.frame, (left, top, width, height))

        self.gaussian_filter.apply(roi_mat, roi_mat, stream=self.stream)

    def convert_bbox(self, bbox: BBox, padding: int, border_width: int):
        left = round(bbox.left) - padding - border_width
        top = round(bbox.top) - padding - border_width

        width = max(round(bbox.width) + 2 * (padding + border_width), 1)
        height = max(round(bbox.height) + 2 * (padding + border_width), 1)

        right = left + width
        bottom = top + height

        left = max(left, 0)
        top = max(top, 0)
        right = min(right, self.max_col)
        bottom = min(bottom, self.max_row)

        return left, top, right, bottom, right - left, bottom - top
