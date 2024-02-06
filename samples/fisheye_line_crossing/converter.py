"""YOLOv7 OBB detector output to bbox converter."""
from typing import Tuple

import numpy as np

from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ObjectModel
from savant.utils.nms import nms_cpu


class TensorToBBoxConverter(BaseObjectModelOutputConverter):
    """YOLOv7 OBB detector output to bbox converter.

    :param confidence_threshold: Select detections with confidence
        greater than specified.
    :param nms_iou_threshold: IoU threshold for NMS.
    :param top_k: Maximum number of output detections.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        top_k: int = 300,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k
        super().__init__()

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Converts detector output layer tensor to bbox tensor.

        Converter is suitable for PyTorch YOLOv8 models.
        Assumed one output layer with shape (batch, 25200, 185) or (25200, 185) for batch=1,
        185 -> (xc,yc,w,h,conf,180*angle categories).
        Outputs best class only for each detection.

        :param output_layers: Output layer tensor list.
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: BBox tensor (class_id, confidence, xc, yc, width, height)
            offset by roi upper left and scaled by roi width and height
        """

        preds = output_layers[0]

        # confidence threshold filter
        keep = preds[:, 4] > self.confidence_threshold
        if not keep.any():
            return np.float32([])
        preds = preds[keep]

        keep = nms_cpu(
            preds[:, :4],
            preds[:, 4],
            self.nms_iou_threshold,
            self.top_k,
        )
        if not keep.any():
            return np.float32([])
        preds = preds[keep]

        confs = preds[:, 4:5]
        # model detects only one class of objects
        class_ids = np.zeros_like(confs)
        xywh = preds[:, :4]

        # roi width / model input width
        ratio_width = roi[2] / model.input.shape[2]
        # roi height / model input height
        ratio_height = roi[3] / model.input.shape[1]
        xywh *= max(ratio_width, ratio_height)
        xywh[:, 0] += roi[0]
        xywh[:, 1] += roi[1]

        angle = np.argmax(preds[:, 5:], axis=1, keepdims=True).astype(np.float32)
        angle -= 90

        bbox_output = np.concatenate((class_ids, confs, xywh, angle), axis=1)

        return bbox_output
