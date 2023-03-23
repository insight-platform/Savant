"""Artist implementation using OpenCV GpuMat."""
from typing import Tuple, Optional, Union, List
from contextlib import AbstractContextManager
import numpy as np
import cv2
from savant.meta.bbox import BBox, RBBox
from .position import Position, get_text_origin


def convert_color(color: Tuple[float, float, float], alpha: int = 255):
    """Convert color from BGR floats to RGBA int8."""
    return int(color[2] * 255), int(color[1] * 255), int(color[0] * 255), alpha


class ArtistGPUMat(AbstractContextManager):
    """Artist implementation using OpenCV GpuMat.

    :param frame: GpuMat header for allocated CUDA-memory of the frame.
    """

    def __init__(self, frame: cv2.cuda.GpuMat) -> None:
        self.stream = cv2.cuda.Stream()
        self.frame: cv2.cuda.GpuMat = frame
        self.width, self.height = self.frame.size()
        self.max_col = self.width - 1
        self.max_row = self.height - 1
        self.alpha_op = cv2.cuda.ALPHA_OVER
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

    @property
    def frame_wh(self):
        return self.width, self.height

    def add_text(
        self,
        text: str,
        anchor_x: int,
        anchor_y: int,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        font_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        border_width: int = 0,
        border_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        bg_color: Optional[Tuple[float, float, float]] = None,
        padding: int = 3,
        anchor_point: Position = Position.CENTER,
    ):
        """Add text on the frame.

        :param text: Display text.
        :param anchor_x: X coordinate of text position.
        :param anchor_y: Y coordinate of text position.
        :param font_scale: Font scale factor that is multiplied by the font-specific base size.
        :param font_thickness: Thickness of the lines used to draw the text.
        :param font_color: Font color, BGR, floats components in range [0;1.0]
        :param border_width: Border width around the text.
        :param border_color: Border color around the text.
        :param bg_color: Background color.
        :param padding: Increase the size of the rectangle around
            the text in each direction, in pixels.
        :param anchor_point: Anchor point of a  rectangle with text.
            For example, if you select Position.CENTER, the rectangle with the text
            will be drawn so that the center of the rectangle is at (x,y).
        """
        self.__init_overlay()

        text_size, baseline = cv2.getTextSize(
            text, self.font_face, font_scale, font_thickness
        )

        text_x, text_y = get_text_origin(
            anchor_point, anchor_x, anchor_y, text_size[0], text_size[1], baseline
        )

        if bg_color or border_width:
            rect_left = text_x - border_width - padding
            rect_top = text_y - text_size[1] - border_width - padding
            rect_right = text_x + text_size[0] + border_width + padding
            rect_bottom = text_y + baseline + border_width + padding

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
            convert_color(font_color),
            font_thickness,
            cv2.LINE_AA,
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

        :param bbox: Bounding box.
        :param border_width:  Border width.
        :param border_color:  Border color.
        :param bg_color: Background color. If None, the rectangle will be transparent.
        :param padding: Increase the size of the rectangle in each direction,
            value in pixels.
        """
        if isinstance(bbox, BBox):
            left, top, right, bottom, _, _ = self.__convert_bbox(
                bbox, padding, border_width
            )

            if bg_color is not None:
                self.frame.colRange(left, right).rowRange(top, bottom).setTo(
                    convert_color(bg_color), stream=self.stream
                )

            if border_width and border_color != bg_color:
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

    def add_rounded_rect(
        self,
        bbox: BBox,
        radius: int,
        bg_color: Tuple[float, float, float],  # BGR
    ):
        self.__init_overlay()

        cv2.rectangle(
            self.overlay,
            (int(bbox.left), int(bbox.top + radius)),
            (int(bbox.right), int(bbox.bottom - radius)),
            convert_color(bg_color),
            -1
        )
        cv2.rectangle(
            self.overlay,
            (int(bbox.left + radius), int(bbox.top)),
            (int(bbox.right - radius), int(bbox.bottom)),
            convert_color(bg_color),
            -1
        )
        # top-left
        cv2.ellipse(
            self.overlay,
            (int(bbox.left + radius), int(bbox.top + radius)),
            (radius, radius),
            180,
            0,
            90,
            convert_color(bg_color),
            -1,
            cv2.LINE_AA,
        )
        # top-right
        cv2.ellipse(
            self.overlay,
            (int(bbox.right - radius), int(bbox.top + radius)),
            (radius, radius),
            270,
            0,
            90,
            convert_color(bg_color),
            -1,
            cv2.LINE_AA,
        )
        # bottom-left
        cv2.ellipse(
            self.overlay,
            (int(bbox.left + radius), int(bbox.bottom - radius)),
            (radius - 1, radius - 1),
            90,
            0,
            90,
            convert_color(bg_color),
            -1,
            cv2.LINE_AA,
        )
        # bottom-right
        cv2.ellipse(
            self.overlay,
            (int(bbox.right - radius), int(bbox.bottom - radius)),
            (radius - 1, radius - 1),
            0,
            0,
            90,
            convert_color(bg_color),
            -1,
            cv2.LINE_AA,
        )

    def add_polygon(
        self,
        vertices: List[Tuple[float, float]],
        line_width: int = 3,
        line_color: Tuple[float, float, float] = (0.0, 0.0, 1.0),  # BGR, Red
        bg_color: Optional[Tuple[float, float, float]] = None,  # BGR
    ):
        """Draw polygon.

        :param vertices: List of points.
        :param line_width: Line width.
        :param line_color: Line color.
        :param bg_color: Background color.
        """
        self.__init_overlay()
        vertices = np.intp(vertices)
        if bg_color is not None:
            cv2.drawContours(
                self.overlay, [vertices], 0, convert_color(bg_color), cv2.FILLED
            )
        if line_width:
            cv2.drawContours(
                self.overlay, [vertices], 0, convert_color(line_color), line_width
            )

    def blur(self, bbox: BBox, padding: int = 0, sigma: Optional[float] = None):
        """Apply gaussian blur to the specified ROI.

        :param bbox: ROI specified as Savant bbox.
        :param padding: Increase the size of the rectangle in each direction,
            value in pixels.
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

        left, top, _, _, width, height = self.__convert_bbox(bbox, padding, 0)
        roi_mat = cv2.cuda.GpuMat(self.frame, (left, top, width, height))

        gaussian_filter.apply(roi_mat, roi_mat, stream=self.stream)

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

    def __convert_bbox(
        self, bbox: BBox, padding: int, border_width: int
    ) -> Tuple[int, int, int, int, int, int]:
        """Convert Savant bbox to OpenCV format.

        :param bbox: Savant BBox structure.
        :param padding: Padding value.
        :param border_width: Box border width.
        :return: Left, top, right, bottom, width, height, clamped to frame dimensions.
        """
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
