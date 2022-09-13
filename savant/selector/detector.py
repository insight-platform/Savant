"""Detector's bbox selectors."""
import numpy as np
from savant.base.selector import BaseSelector


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


class BBoxSelector(BaseSelector):
    """Detector bbox per class selector.

    .. todo::
        - support max_width/max_height
        - support topk
        - check width/height before nms?

    :param confidence_threshold: confidence threshold
    :param nms_iou_threshold: nms iou threshold
    :param min_width: minimal bbox width
    :param min_height: minimal bbox height
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        min_width: int = 0,
        min_height: int = 0,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.min_width = min_width
        self.min_height = min_height
        super().__init__()

    def __call__(self, bbox_tensor: np.ndarray) -> np.ndarray:
        """Filters bboxes by confidence and size, applies NMS.

        :param bbox_tensor: tensor(class_id, confidence, left, top, width, height)
        :return: Selected bbox tensor
        """
        selected_bbox_tensor = bbox_tensor.copy()

        if self.confidence_threshold:
            selected_bbox_tensor = selected_bbox_tensor[
                selected_bbox_tensor[:, 1] > self.confidence_threshold
            ]

        if self.nms_iou_threshold:
            keep = nms_cpu(
                selected_bbox_tensor[:, 2:6],
                selected_bbox_tensor[:, 1],
                self.nms_iou_threshold,
            )
            selected_bbox_tensor = selected_bbox_tensor[keep == 1]

        if self.min_width:
            selected_bbox_tensor = selected_bbox_tensor[
                selected_bbox_tensor[:, 4] >= self.min_width
            ]

        if self.min_height:
            selected_bbox_tensor = selected_bbox_tensor[
                selected_bbox_tensor[:, 5] >= self.min_height
            ]

        return selected_bbox_tensor
