"""Utility for scale rotated bboxes."""
import numpy as np


def scale_rbbox(
    bboxes: np.ndarray, scale_factor_x: float, scale_factor_y: float
) -> np.ndarray:
    """Scaling rotated boxes.

    :param bboxes: np array of bboxes, shape Nx5. Row is [cx, cy, w, h, angle]
    :param scale_factor_x: scale factor for x coordinates
    :param scale_factor_y: scale factor for y coordinates
    """
    bboxes_zero_angle = bboxes[np.mod(bboxes[:, 4], 90) == 0]
    bboxes_not_zero_angle = bboxes[np.mod(bboxes[:, 4], 90) != 0]

    if bboxes_not_zero_angle.shape[0] > 0:
        scale_x = np.array([scale_factor_x] * bboxes_not_zero_angle.shape[0])
        scale_y = np.array([scale_factor_y] * bboxes_not_zero_angle.shape[0])
        scale_x_2 = scale_x * scale_x
        scale_y_2 = scale_y * scale_y
        cotan = 1 / np.tan(bboxes_not_zero_angle[:, 4] / 180 * np.pi)
        cotan_2 = cotan * cotan
        scale_angle = np.arccos(
            scale_x
            * np.sign(bboxes_not_zero_angle[:, 4])
            / np.sqrt(scale_x_2 + scale_y_2 * cotan * cotan)
        )
        nscale_height = np.sqrt(scale_x_2 + scale_y_2 * cotan_2) / np.sqrt(1 + cotan_2)
        ayh = 1 / np.tan((90 - bboxes_not_zero_angle[:, 4]) / 180 * np.pi)
        nscale_width = np.sqrt(scale_x_2 + scale_y_2 * ayh * ayh) / np.sqrt(
            1 + ayh * ayh
        )
        bboxes_not_zero_angle[:, 4] = 90 - (scale_angle * 180) / np.pi
        bboxes_not_zero_angle[:, 3] = bboxes_not_zero_angle[:, 3] * nscale_height
        bboxes_not_zero_angle[:, 2] = bboxes_not_zero_angle[:, 2] * nscale_width
        bboxes_not_zero_angle[:, 1] = bboxes_not_zero_angle[:, 1] * scale_y
        bboxes_not_zero_angle[:, 0] = bboxes_not_zero_angle[:, 0] * scale_x

    if bboxes_zero_angle.shape[0] > 0:
        bboxes_zero_angle[:, 3] = bboxes_zero_angle[:, 3] * scale_factor_y
        bboxes_zero_angle[:, 2] = bboxes_zero_angle[:, 2] * scale_factor_x
        bboxes_zero_angle[:, 1] = bboxes_zero_angle[:, 1] * scale_factor_y
        bboxes_zero_angle[:, 0] = bboxes_zero_angle[:, 0] * scale_factor_x

    return np.concatenate([bboxes_zero_angle, bboxes_not_zero_angle])
