"""Detector's bbox selectors."""
from numba import njit, uint8, uint16, float32
import numpy as np
from savant.base.selector import BaseSelector


@njit(uint8[:](float32[:, :], float32[:], float32), nogil=True)
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

    keep = np.zeros((len(bboxes),), dtype=np.uint8)
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


@njit(float32[:, :](float32[:, :], uint16, uint16, uint16, uint16), nogil=True)
def min_max_bbox_size_selector(
    bbox_tensor: np.ndarray,
    min_width: int = 0,
    min_height: int = 0,
    max_width: int = 0,
    max_height: int = 0,
) -> np.ndarray:
    """Filters bboxes by size.

    :param bbox_tensor: tensor(class_id, confidence, left, top, width, height)
    :param min_width: minimal bbox width
    :param min_height: minimal bbox height
    :param max_width: maximum bbox width
    :param max_height: maximum bbox height
    :return: Selected bbox tensor
    """
    selected_bbox_tensor = bbox_tensor.copy()

    if min_width:
        selected_bbox_tensor = selected_bbox_tensor[
            selected_bbox_tensor[:, 4] > min_width
        ]

    if min_height:
        selected_bbox_tensor = selected_bbox_tensor[
            selected_bbox_tensor[:, 5] > min_height
        ]

    if max_width:
        selected_bbox_tensor = selected_bbox_tensor[
            selected_bbox_tensor[:, 4] < max_width
        ]

    if max_height:
        selected_bbox_tensor = selected_bbox_tensor[
            selected_bbox_tensor[:, 5] < max_height
        ]

    return selected_bbox_tensor


class MinMaxSizeBBoxSelector(BaseSelector):
    """Detector bbox size selector.

    :param min_width: minimal bbox width
    :param min_height: minimal bbox height
    :param max_width: maximum bbox width
    :param max_height: maximum bbox height
    """

    def __init__(
        self,
        min_width: int = 0,
        min_height: int = 0,
        max_width: int = 0,
        max_height: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height

    def __call__(self, bbox_tensor: np.ndarray) -> np.ndarray:
        """Filters bboxes by confidence and size, applies NMS.

        :param bbox_tensor: tensor(class_id, confidence, left, top, width, height)
        :return: Selected bbox tensor
        """
        return min_max_bbox_size_selector(
            bbox_tensor=bbox_tensor,
            min_width=self.min_width,
            min_height=self.min_height,
            max_width=self.max_width,
            max_height=self.max_height,
        )


@njit(
    float32[:, :](float32[:, :], float32, float32, uint16, uint16, uint16, uint16),
    nogil=True,
)
def default_selector(
    bbox_tensor: np.ndarray,
    confidence_threshold: float = 0.0,
    nms_iou_threshold: float = 0.0,
    min_width: int = 0,
    min_height: int = 0,
    max_width: int = 0,
    max_height: int = 0,
) -> np.ndarray:
    """Filters bboxes by confidence and size, applies NMS.

    :param bbox_tensor: tensor(class_id, confidence, left, top, width, height)
    :param confidence_threshold: confidence threshold
    :param nms_iou_threshold: nms iou threshold
    :param min_width: minimal bbox width
    :param min_height: minimal bbox height
    :param max_width: maximum bbox width
    :param max_height: maximum bbox height
    :return: Selected bbox tensor
    """
    selected_bbox_tensor = bbox_tensor.copy()

    if confidence_threshold:
        selected_bbox_tensor = selected_bbox_tensor[
            selected_bbox_tensor[:, 1] > confidence_threshold
        ]

    if min_width or min_height or max_width or max_height:
        selected_bbox_tensor = min_max_bbox_size_selector(
            selected_bbox_tensor,
            min_width=min_width,
            min_height=min_height,
            max_width=max_width,
            max_height=max_height,
        )

    if nms_iou_threshold:
        keep = nms_cpu(
            selected_bbox_tensor[:, 2:6],
            selected_bbox_tensor[:, 1],
            nms_iou_threshold,
        )
        selected_bbox_tensor = selected_bbox_tensor[keep == 1]

    return selected_bbox_tensor


class BBoxSelector(BaseSelector):
    """Detector bbox per class selector.

    :param confidence_threshold: confidence threshold
    :param nms_iou_threshold: nms iou threshold
    :param min_width: minimal bbox width
    :param min_height: minimal bbox height
    :param max_width: maximum bbox width
    :param max_height: maximum bbox height
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        min_width: int = 0,
        min_height: int = 0,
        max_width: int = 0,
        max_height: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.min_width = min_width
        self.min_height = min_height
        self.max_width = max_width
        self.max_height = max_height

    def __call__(self, bbox_tensor: np.ndarray) -> np.ndarray:
        """Filters bboxes by confidence and size, applies NMS.

        :param bbox_tensor: tensor(class_id, confidence, left, top, width, height)
        :return: Selected bbox tensor
        """
        return default_selector(
            bbox_tensor=bbox_tensor,
            confidence_threshold=self.confidence_threshold,
            nms_iou_threshold=self.nms_iou_threshold,
            min_width=self.min_width,
            min_height=self.min_height,
            max_width=self.max_width,
            max_height=self.max_height,
        )
