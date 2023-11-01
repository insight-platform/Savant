"""Artist implementation using OpenCV GpuMat."""
from contextlib import AbstractContextManager
from typing import List, Optional, Tuple, Union

import numpy as np

import cv2
from savant_rs.draw_spec import PaddingDraw
from savant_rs.primitives.geometry import BBox, RBBox

from .position import Position, get_bottom_left_point


class ArtistGPUMat(AbstractContextManager):
    """Artist implementation using OpenCV GpuMat.

    :param frame: GpuMat header for allocated CUDA-memory of the frame.
    """

    def __init__(self, frame: cv2.cuda.GpuMat, stream: cv2.cuda.Stream) -> None:
        self.stream = stream
        self.frame: cv2.cuda.GpuMat = frame
        self.width, self.height = self.frame.size()
        self.max_col = self.width - 1
        self.max_row = self.height - 1
        self.alpha_op = cv2.cuda.ALPHA_OVER
        self.overlay = None
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.gaussian_filter = None

    def __exit__(self, *exc_details):
        # apply alpha comp if overlay is not null
        if self.overlay is not None:
            overlay_gpu = cv2.cuda.GpuMat(self.height, self.width, cv2.CV_8UC4)
            overlay_gpu.upload(self.overlay, self.stream)
            cv2.cuda.alphaComp(
                overlay_gpu, self.frame, self.alpha_op, self.frame, stream=self.stream
            )

    @property
    def frame_wh(self):
        return self.width, self.height

    def add_text(
        self,
        text: str,
        anchor: Tuple[int, int],
        font_scale: float = 0.5,
        font_thickness: int = 1,
        font_color: Tuple[int, int, int, int] = (255, 255, 255, 255),  # white
        border_width: int = 0,
        border_color: Tuple[int, int, int, int] = (255, 0, 0, 255),  # red
        bg_color: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 255),  # black
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        anchor_point_type: Position = Position.CENTER,
    ) -> int:
        """Draw text, text backround box and text background box border on the frame.
        Does not draw anything if text is empty.

        :param text: Display text.
        :param anchor: X,Y coordinates of text position.
        :param font_scale: Font scale factor that is multiplied by the font-specific base size.
        :param font_thickness: Thickness of the lines used to draw the text, >= 0.
        :param font_color: Font color, RGBA, ints in range [0;255].
        :param border_width: Border width around the text.
        :param border_color: Border color around the text, RGBA, ints in range [0;255].
        :param bg_color: Background color, RGBA, ints in range [0;255].
        :param padding: Increase the size of the rectangle around
            the text in each direction (left, top, right, bottom), in pixels.
        :param anchor_point_type: Anchor point of a  rectangle with text.
            For example, if you select Position.CENTER, the rectangle with the text
            will be drawn so that the center of the rectangle is at (x,y).
        :return: Text box height, even if nothing was drawn.
        """
        draw_text = font_scale > 0 and len(text) > 0 and font_color[3] > 0
        draw_border = border_width > 0 and border_color[3] > 0
        draw_bg = bg_color is not None and bg_color[3] > 0

        text_size, baseline = cv2.getTextSize(
            text, self.font_face, font_scale, font_thickness
        )

        if len(text) > 0:
            text_left, text_bottom = get_bottom_left_point(
                anchor_point_type, anchor, text_size, baseline
            )

            self.__init_overlay()

            rect_left = text_left - border_width - padding[0]
            rect_top = text_bottom - text_size[1] - border_width - padding[1]
            rect_right = text_left + text_size[0] + border_width + padding[2]
            rect_bottom = text_bottom + baseline + border_width + padding[3]

            rect_tl = rect_left, rect_top
            rect_br = rect_right, rect_bottom

            if draw_bg:
                cv2.rectangle(self.overlay, rect_tl, rect_br, bg_color, cv2.FILLED)

            if draw_border:
                cv2.rectangle(
                    self.overlay,
                    rect_tl,
                    rect_br,
                    border_color,
                    border_width,
                )

            if draw_text:
                cv2.putText(
                    self.overlay,
                    text,
                    (text_left, text_bottom),
                    self.font_face,
                    font_scale,
                    font_color,
                    font_thickness,
                    cv2.LINE_AA,
                )
        return text_size[1] + baseline

    # pylint:disable=too-many-arguments
    def add_bbox(
        self,
        bbox: Union[BBox, RBBox],
        border_width: int = 3,
        border_color: Tuple[int, int, int, int] = (0, 255, 0, 255),  # RGBA, Green
        bg_color: Optional[Tuple[int, int, int, int]] = None,  # RGBA
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        """Draw bbox on frame.

        :param bbox: Bounding box.
        :param border_width:  Border width.
        :param border_color:  Border color, RGBA, ints in range [0;255].
        :param bg_color: Background color, RGBA, ints in range [0;255].
            If None, the rectangle will be transparent.
        :param padding: Increase the size of the rectangle in each direction,
            value in pixels, tuple of 4 values (left, top, right, bottom).
        """
        draw_border = border_width > 0 and border_color[3] > 0
        draw_bg = bg_color is not None and bg_color[3] > 0
        if not draw_border and not draw_bg:
            return

        if isinstance(bbox, BBox):
            left, top, right, bottom = bbox.visual_box(
                PaddingDraw(*padding), border_width, self.max_col, self.max_row
            ).as_ltrb_int()
            if draw_bg:
                self.frame.colRange(left, right).rowRange(top, bottom).setTo(
                    bg_color, stream=self.stream
                )

            if draw_border and (border_color != bg_color or not draw_bg):
                self.frame.colRange(left, right).rowRange(
                    top, top + border_width
                ).setTo(border_color, stream=self.stream)
                self.frame.colRange(left, right).rowRange(
                    bottom - border_width, bottom
                ).setTo(border_color, stream=self.stream)
                self.frame.colRange(left, left + border_width).rowRange(
                    top, bottom
                ).setTo(border_color, stream=self.stream)
                self.frame.colRange(right - border_width, right).rowRange(
                    top, bottom
                ).setTo(border_color, stream=self.stream)

        elif isinstance(bbox, RBBox):
            padded = bbox.new_padded(PaddingDraw(*padding))
            self.add_polygon(
                padded.vertices_int,
                border_width,
                border_color,
                bg_color,
            )

    def add_rounded_rect(
        self,
        bbox: BBox,
        radius: int,
        bg_color: Tuple[int, int, int, int],  # RGBA
    ):
        """Draw rounded rect.

        :param bbox: Bounding box.
        :param radius: Border radius, in px.
        :param bg_color: Background color, RGBA, ints in range [0;255].
        """
        if bg_color[3] <= 0:
            return

        self.__init_overlay()

        cv2.rectangle(
            self.overlay,
            (int(bbox.left), int(bbox.top + radius)),
            (int(bbox.right), int(bbox.bottom - radius)),
            bg_color,
            cv2.FILLED,
        )
        cv2.rectangle(
            self.overlay,
            (int(bbox.left + radius), int(bbox.top)),
            (int(bbox.right - radius), int(bbox.bottom)),
            bg_color,
            cv2.FILLED,
        )

        # rounded corners: center(x, y), rotation angle
        rounded_corners = [
            ((int(bbox.left + radius), int(bbox.top + radius)), 180),  # left-top
            ((int(bbox.right - radius), int(bbox.top + radius)), 270),  # right-top
            ((int(bbox.left + radius), int(bbox.bottom - radius)), 90),  # left-bottom
            ((int(bbox.right - radius), int(bbox.bottom - radius)), 0),  # right-bottom
        ]
        for center, angle in rounded_corners:
            cv2.ellipse(
                self.overlay,
                center,
                (radius, radius),
                angle,
                0,
                90,
                bg_color,
                cv2.FILLED,
                cv2.LINE_AA,
            )

    def add_circle(
        self,
        center: Tuple[int, int],
        radius: int,
        color: Tuple[int, int, int, int],
        thickness: int,
        line_type: int = cv2.LINE_AA,
    ):
        """Draw circle.

        :param center: Circle center.
        :param radius: Circle radius.
        :param color: Circle line color, RGBA, ints in range [0;255].
        :param thickness: Circle line thickness.
        :param line_type: Circle line type.
        """
        if color[3] <= 0 or (thickness <= 0 and radius <= 0):
            return
        self.__init_overlay()
        cv2.circle(self.overlay, center, radius, color, thickness, line_type)

    def add_line(
        self,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int, int] = (255, 0, 0, 255),  # RGBA, Red,
        thickness: int = 3,
        type: int = cv2.LINE_AA,
    ):
        """Draw line.

        :param pt1: First point.
        :param pt2: Second point.
        :param color: Line color, RGBA, ints in range [0;255].
        :param thickness: Line thickness.
        :param type: Line type.
        """
        if color[3] <= 0 or thickness <= 0:
            return
        self.__init_overlay()
        cv2.line(self.overlay, pt1, pt2, color, thickness, type)

    def add_polygon(
        self,
        vertices: List[Tuple[int, int]],
        line_width: int = 3,
        line_color: Tuple[int, int, int, int] = (255, 0, 0, 255),  # RGBA, Red
        bg_color: Optional[Tuple[int, int, int, int]] = None,  # RGBA
    ):
        """Draw polygon.

        :param vertices: List of points.
        :param line_width: Line width.
        :param line_color: Line color, RGBA, ints in range [0;255].
        :param bg_color: Background color, RGBA, ints in range [0;255].
        """
        draw_contour = line_width > 0 and line_color[3] > 0
        draw_fill = bg_color is not None and bg_color[3] > 0
        if not draw_contour and not draw_fill:
            return

        self.__init_overlay()
        vertices = np.array(vertices)[np.newaxis, ...]
        if draw_fill:
            cv2.drawContours(self.overlay, vertices, 0, bg_color, cv2.FILLED)
        if draw_contour and (not draw_fill or line_color != bg_color):
            cv2.drawContours(self.overlay, vertices, 0, line_color, line_width)

    def blur(
        self,
        bbox: BBox,
        padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
        sigma: Optional[float] = None,
    ):
        """Apply gaussian blur to the specified ROI.

        :param bbox: ROI specified as Savant bbox.
        :param padding: Increase the size of the rectangle in each direction,
            value in pixels, left, top, right, bottom.
        :param sigma: gaussian blur stddev.
        """
        if sigma is None:
            sigma = min(bbox.width, bbox.height) / 10

        radius = int(sigma * 4 + 0.5)
        if radius % 2 == 0:
            radius += 1
        radius = max(radius, 1)
        radius = min(radius, 31)

        gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC4, cv2.CV_8UC4, (radius, radius), sigma
        )

        left, top, width, height = bbox.visual_box(
            PaddingDraw(*padding), 0, self.max_col, self.max_row
        ).as_ltwh_int()
        roi_mat = cv2.cuda.GpuMat(self.frame, (left, top, width, height))

        gaussian_filter.apply(roi_mat, roi_mat, stream=self.stream)

    def copy_frame_region(
        self, bbox: BBox, padding: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> cv2.cuda_GpuMat:
        """Copy a region of the frame to a new GpuMat.

        :param bbox: ROI specified as Savant bbox.
        :return: GpuMat with the specified region.
        """
        left, top, width, height = bbox.visual_box(
            PaddingDraw(*padding), 0, self.max_col, self.max_row
        ).as_ltwh_int()
        roi_mat = cv2.cuda_GpuMat(self.frame, (left, top, width, height))
        return roi_mat

    def add_graphic(self, img: cv2.cuda.GpuMat, origin: Tuple[int, int]):
        """Overlays an image onto the frame, e.g. a logo.

        :param img: RGBA image in GPU memory
        :param origin: Coordinates of left top corner of img in frame space. (left, top)
        """
        frame_left, frame_top = origin
        if frame_left >= self.width or frame_top >= self.height:
            return

        img_w, img_h = img.size()
        if frame_left + img_w < 0 or frame_top + img_h < 0:
            return

        if frame_left < 0:
            img_left = abs(frame_left)
        else:
            img_left = 0

        if frame_top < 0:
            img_top = abs(frame_top)
        else:
            img_top = 0

        frame_right = frame_left + img_w
        frame_bottom = frame_top + img_h

        if frame_right >= self.width:
            img_right = self.width - frame_left
        else:
            img_right = img_w

        if frame_bottom >= self.height:
            img_bottom = self.height - frame_top
        else:
            img_bottom = img_h

        frame_left = max(frame_left, 0)
        frame_top = max(frame_top, 0)
        frame_right = min(frame_right, self.width)
        frame_bottom = min(frame_bottom, self.height)

        frame_roi = self.frame.colRange(frame_left, frame_right).rowRange(
            frame_top, frame_bottom
        )
        img_roi = img.colRange(img_left, img_right).rowRange(img_top, img_bottom)

        img_roi.copyTo(self.stream, frame_roi)

    def __init_overlay(self):
        """Init overlay image."""
        if self.overlay is None:
            self.overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)
