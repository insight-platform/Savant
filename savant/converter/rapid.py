"""Tensor to bounding box converter."""
from typing import Tuple
import numpy as np
from pysavantboost import nms
from savant.converter.scale import scale_rbbox
from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ObjectModel


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
            return np.empty((0, 7))

        bboxes = nms(
            output_layers[0],
            self.nms_iou_threshold,
            self.confidence_threshold,
            self.top_k,
        )

        if not bboxes.shape[0]:
            return np.empty((0, 7))

        roi_left, roi_top, roi_width, roi_height = roi

        original_bboxes = scale_rbbox(
            bboxes[:, :5],
            roi_width,
            roi_height,
        )

        # correct xc, yc
        original_bboxes[:, 0] += roi_left
        original_bboxes[:, 1] += roi_top

        return np.concatenate(
            [
                np.zeros((original_bboxes.shape[0], 1), dtype=np.intc),
                np.expand_dims(bboxes[:, 5], 1),
                original_bboxes[:, :],
            ],
            axis=1,
        )
