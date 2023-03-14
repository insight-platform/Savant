from typing import Tuple
import numpy as np
import cv2


def load_sprite(path: str, target_height: int) -> cv2.cuda.GpuMat:
    """Read from file, resize according to target height, move to GPU."""
    sprite = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    sprite = cv2.cvtColor(sprite, cv2.COLOR_BGRA2RGBA)
    resize_coeff = target_height / sprite.shape[0]
    sprite = cv2.resize(
        sprite,
        dsize=None,
        fx=resize_coeff,
        fy=resize_coeff,
        interpolation=cv2.INTER_CUBIC,
    )
    return cv2.cuda.GpuMat(sprite)


def get_font_scale(
    target_height_px: int,
    font_thickness: float = 1,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    sample_text: str = '0123456789',
    font_scale_range: Tuple[float, float] = (0.1, 5),
    font_scale_step: float = 0.1,
) -> float:
    """Find a font scale for OpenCV's text according to target text height in pixels."""

    font_scale = font_scale_range[0]
    text_size, _ = cv2.getTextSize(sample_text, font_face, font_scale, font_thickness)
    min_delta = abs(target_height_px - text_size[1])

    scale_range_start = font_scale + font_scale_step
    for scale in np.arange(scale_range_start, font_scale_range[1], font_scale_step):
        text_size, baseline = cv2.getTextSize(
            sample_text, font_face, scale, font_thickness
        )
        delta = abs(target_height_px - text_size[1])
        if delta < min_delta:
            min_delta = delta
            font_scale = scale
            _ = baseline
    return font_scale
