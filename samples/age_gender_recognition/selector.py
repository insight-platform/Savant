"""Detector's bbox selectors."""
from numba import njit, uint16, float32
import numpy as np
from savant.base.selector import BaseSelector


@njit(
    float32[:, :](float32[:, :], uint16, uint16, uint16, uint16),
    nogil=True,
)
def selector(
    bbox_tensor: np.ndarray,
    min_width: int = 0,
    min_height: int = 0,
    max_width: int = 0,
    max_height: int = 0,
) -> np.ndarray:
    """Filters bboxes by confidence and size, applies NMS.

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
        return selector(
            bbox_tensor=bbox_tensor,
            min_width=self.min_width,
            min_height=self.min_height,
            max_width=self.max_width,
            max_height=self.max_height,
        )
