"""Tensor to bounding box converter."""

from typing import Tuple

import numpy as np
from numba import float32, njit, void
from pysavantboost import nms

from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ObjectModel


@njit(void(float32[:, :], float32, float32), nogil=True)
def scale_rbbox(
    bboxes: np.ndarray, scale_factor_x: float, scale_factor_y: float
) -> np.ndarray:
    """Scaling rotated boxes in-place.

    :param bboxes: np array of bboxes, shape Nx5. Row is [cx, cy, w, h, angle]
    :param scale_factor_x: scale factor for x coordinates
    :param scale_factor_y: scale factor for y coordinates
    """
    no_angle_mask = np.mod(bboxes[:, 4], 90) == 0
    angle_mask = ~no_angle_mask

    if np.any(angle_mask):
        scale_x_2 = scale_factor_x * scale_factor_x
        scale_y_2 = scale_factor_y * scale_factor_y
        cotan = 1 / np.tan(bboxes[angle_mask, 4] / 180 * np.pi)
        cotan_2 = cotan * cotan
        scale_angle = np.arccos(
            scale_factor_x
            * np.sign(bboxes[angle_mask, 4])
            / np.sqrt(scale_x_2 + scale_y_2 * cotan_2)
        )
        nscale_height = np.sqrt((scale_x_2 + scale_y_2 * cotan_2) / (1 + cotan_2))
        ayh = 1 / np.tan((90 - bboxes[angle_mask, 4]) / 180 * np.pi)
        ayh_2 = ayh * ayh
        nscale_width = np.sqrt((scale_x_2 + scale_y_2 * ayh_2) / (1 + ayh * ayh))
        bboxes[angle_mask, 4] = 90 - (scale_angle * 180) / np.pi
        bboxes[angle_mask, 3] *= nscale_height
        bboxes[angle_mask, 2] *= nscale_width
        bboxes[angle_mask, 1] *= scale_factor_y
        bboxes[angle_mask, 0] *= scale_factor_x

    if np.any(no_angle_mask):
        bboxes[no_angle_mask, 0:3:2] *= scale_factor_x
        bboxes[no_angle_mask, 1:4:2] *= scale_factor_y


class TensorToBBoxConverter(BaseObjectModelOutputConverter):
    """`RAPiD <https://github.com/duanzhiihao/RAPiD>`_ output to bbox
    converter.

    :param confidence_threshold: confidence threshold (pre-cluster-threshold)
    :param nms_iou_threshold: nms iou threshold
    :param top_k: leave no more than top K bboxes with maximum confidence
    """

    def __init__(
        self,
        confidence_threshold: float = 0.4,
        nms_iou_threshold: float = 0.5,
        top_k: int = 70,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k
        super().__init__()

        # this fake nms function call is made to preload the library and
        # avoid the delay on the first processing frames
        _ = nms(np.random.rand(2, 6).astype(np.float32), 0.5, 0.4, 70)

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Converts RAPiD detector output layer tensor to bbox tensor.

        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: BBox tensor (class_id, confidence, xc, yc, width, height, angle)
            in roi scale
        """
        if not output_layers[0].shape[0]:
            return np.empty((0, 7), dtype=np.float32)

        bboxes = nms(
            output_layers[0],
            self.nms_iou_threshold,
            self.confidence_threshold,
            self.top_k,
        )

        if not bboxes.shape[0]:
            return np.empty((0, 7), dtype=np.float32)

        roi_left, roi_top, roi_width, roi_height = roi

        original_bboxes = bboxes[:, :5].copy()
        scale_rbbox(
            original_bboxes,
            roi_width,
            roi_height,
        )
        # correct xc, yc
        original_bboxes[:, 0] += roi_left
        original_bboxes[:, 1] += roi_top

        return np.concatenate(
            [
                np.zeros((original_bboxes.shape[0], 1), dtype=np.float32),
                np.expand_dims(bboxes[:, 5], 1),
                original_bboxes[:, :],
            ],
            axis=1,
        )
