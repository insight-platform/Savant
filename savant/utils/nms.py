"""Non-max suppression (NMS) implementation.

We can't use the same code with different array modules (cupy, numpy) because of
the numba limitation "Cannot determine Numba type of <class 'function'>".

TODO: Compare with chainer implementation.
    https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/non_maximum_suppression.py
"""
import cupy as cp
import numba as nb
import numpy as np

__all__ = ['nms_cpu', 'nms_gpu']


@nb.njit('u2[:](f4[:, :], f4[:], f4)', nogil=True)
def nms_cpu(
    bboxes: np.ndarray, confidences: np.ndarray, threshold: float
) -> np.ndarray:
    """NumPy NMS."""
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

        indices = np.where(over <= threshold)[0]
        order = order[indices + 1]

    return keep


def nms_gpu(
    bboxes: cp.ndarray, confidences: cp.ndarray, threshold: float
) -> cp.ndarray:
    """CuPy NMS."""
    x_left = bboxes[:, 0]
    y_top = bboxes[:, 1]
    x_right = bboxes[:, 0] + bboxes[:, 2]
    y_bottom = bboxes[:, 1] + bboxes[:, 3]

    areas = (x_right - x_left) * (y_bottom - y_top)
    order = confidences.argsort()[::-1]

    keep = cp.zeros((len(bboxes),), dtype=cp.uint16)
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep[idx_self] = 1

        xx1 = cp.maximum(x_left[idx_self], x_left[idx_other])
        yy1 = cp.maximum(y_top[idx_self], y_top[idx_other])
        xx2 = cp.minimum(x_right[idx_self], x_right[idx_other])
        yy2 = cp.minimum(y_bottom[idx_self], y_bottom[idx_other])

        width = cp.maximum(0.0, xx2 - xx1)
        height = cp.maximum(0.0, yy2 - yy1)
        inter = width * height

        over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        indices = cp.where(over <= threshold)[0]
        order = order[indices + 1]

    return keep
