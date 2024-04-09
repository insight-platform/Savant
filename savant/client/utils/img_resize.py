"""Image resizing utilities."""

import math
from typing import Tuple

import cv2
import numpy as np


def pad_to_aspect(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Ensure that the image has the given aspect ratio by adding padding
    without resizing the image contents.

    :param img: Image to pad.
    :param target_size: Target size (img_w, img_h).
    :return: Padded image.
    """
    target_w, target_h = target_size
    img_h, img_w, img_c = img.shape

    assert target_w >= img_w and target_h >= img_h

    if target_h > img_h:
        top = (target_h - img_h) // 2
        bottom = target_h - img_h - top
    else:
        top = bottom = 0
    if target_w > img_w:
        left = (target_w - img_w) // 2
        right = target_w - img_w - left
    else:
        left = right = 0

    black_color = [0] * img_c
    if img_c == 4:
        black_color[3] = 255

    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=black_color
    )


def resize_preserving_aspect(
    img: np.ndarray, target_size: Tuple[int, int]
) -> np.ndarray:
    """Resize image while preserving aspect ratio of the image contents.

    :param img: Image to resize.
    :param target_size: Target size (img_w, img_h).
    :return: Resized image.
    """
    target_w, target_h = target_size
    img_h, img_w, img_c = img.shape

    if (img_h == target_h and img_w < target_w) or (
        img_w == target_w and img_h < target_h
    ):
        # it's enough to add padding either on top and bottom or on left and right
        return pad_to_aspect(img, target_size)

    img_aspect = img_w / img_h
    target_aspect = target_w / target_h

    if math.isclose(img_aspect, target_aspect, rel_tol=1e-3):
        # aspect ratios are close enough
        img = cv2.resize(img, (target_w, target_h))
    else:
        new_img = np.zeros((target_h, target_w, img_c), dtype=np.uint8)
        if img_c == 4:
            new_img[:, :, 3] = 255

        if img_aspect > target_aspect:
            # add padding on top and bottom
            # and possibly resize to match the target img_w
            if img_w != target_w:
                # resize so that the img_w matches the target img_w
                # while preserving aspect ratio
                resized_w = target_w
                resized_h = round(img_h * target_w / img_w)
                resized = cv2.resize(img, (resized_w, resized_h))
            else:
                # img_w matches, no need to resize
                resized_w = img_w
                resized_h = img_h
                resized = img
            top = (target_h - resized_h) // 2
            bottom = top + resized_h
            new_img[top:bottom, :, :] = resized
        else:
            # add padding on left and right
            # and possibly resize to match the target img_h
            if img_h != target_h:
                # resize so that the img_h matches the target img_h
                # while preserving aspect ratio
                resized_h = target_h
                resized_w = round(img_w * target_h / img_h)
                resized = cv2.resize(img, (resized_w, resized_h))
            else:
                resized_h = img_h
                resized_w = img_w
                resized = img
            left = (target_w - resized_w) // 2
            right = left + resized_w
            new_img[:, left:right, :] = resized

        img = new_img
    return img
