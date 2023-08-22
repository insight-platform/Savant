import numba as nb
import numpy as np


@nb.njit('u2[:](f4[:, :], f4[:], f4)', nogil=True)
def nms_cpu(
    bboxes: np.ndarray, confidences: np.ndarray, threshold: float
) -> np.ndarray:
    """Non-max suppression."""
    x_left = bboxes[:, 0]
    y_top = bboxes[:, 1]
    x_right = bboxes[:, 0] + bboxes[:, 2]
    y_bottom = bboxes[:, 1] + bboxes[:, 3]

    areas = (x_right - x_left) * (y_bottom - y_top)
    order = confidences.argsort()[::-1]

    keep = np.zeros((len(bboxes),), dtype=np.uint16)
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep[idx_self] = 1

        xx1 = np.maximum(x_left[idx_self], x_left[idx_other])
        yy1 = np.maximum(y_top[idx_self], y_top[idx_other])
        xx2 = np.minimum(x_right[idx_self], x_right[idx_other])
        yy2 = np.minimum(y_bottom[idx_self], y_bottom[idx_other])

        width = np.maximum(0.0, xx2 - xx1)
        height = np.maximum(0.0, yy2 - yy1)
        inter = width * height

        over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= threshold)[0]
        order = order[inds + 1]

    return keep
