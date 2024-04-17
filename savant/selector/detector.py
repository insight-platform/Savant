"""Detector's bbox selectors."""

import numba as nb
import numpy as np

from savant.base.selector import BaseSelector
from savant.utils.nms import nms_cpu


@nb.njit('f4[:, :](f4[:, :], u2, u2, u2, u2)', nogil=True, cache=True)
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
        **kwargs,
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


@nb.njit('f4[:, :](f4[:, :], f4, f4, u2, u2, u2, u2, u2)', nogil=True, cache=True)
def default_selector(
    bbox_tensor: np.ndarray,
    confidence_threshold: float = 0.0,
    nms_iou_threshold: float = 0.0,
    top_k: int = 0,
    min_width: int = 0,
    min_height: int = 0,
    max_width: int = 0,
    max_height: int = 0,
) -> np.ndarray:
    """Filters bboxes by confidence and size, applies NMS.

    :param bbox_tensor: tensor(class_id, confidence, left, top, width, height)
    :param confidence_threshold: confidence threshold
    :param nms_iou_threshold: nms iou threshold
    :param top_k: top k bboxes to keep
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
            # should specify default with numba.njit
            top_k if top_k > 0 else selected_bbox_tensor.shape[0],
        )
        selected_bbox_tensor = selected_bbox_tensor[keep]

    return selected_bbox_tensor


class BBoxSelector(BaseSelector):
    """Detector bbox per class selector.

    :param confidence_threshold: confidence threshold
    :param nms_iou_threshold: nms iou threshold
    :param top_k: top k bboxes to keep
    :param min_width: minimal bbox width
    :param min_height: minimal bbox height
    :param max_width: maximum bbox width
    :param max_height: maximum bbox height
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        top_k: int = 0,
        min_width: int = 0,
        min_height: int = 0,
        max_width: int = 0,
        max_height: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k
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
            top_k=self.top_k,
            min_width=self.min_width,
            min_height=self.min_height,
            max_width=self.max_width,
            max_height=self.max_height,
        )
