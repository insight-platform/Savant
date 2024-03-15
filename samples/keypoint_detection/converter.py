"""YOLOv8Pose postprocessing (converter)."""
from typing import Any, Tuple

import numpy as np
from numba.typed import List
from savant.base.converter import BaseComplexModelOutputConverter
from savant.base.model import ComplexModel
from savant.utils.nms import nms_cpu


class YoloV8PoseConverter(BaseComplexModelOutputConverter):
    """`YOLOv8Pose converter."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        nms_iou_threshold: float = 0.45,
        top_k: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ComplexModel,
        roi: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, List[List[Tuple[str, Any, float]]]]:
        """Converts output layer tensors to bboxes and key points.

        :param output_layers: Output layers tensor
        :param model: Model definition, required parameters: input tensor shape
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: a combination of :py:class:`.BaseObjectModelOutputConverter` and
            :py:class:`.BaseAttributeModelOutputConverter` outputs:

            * BBox tensor ``(class_id, confidence, xc, yc, width, height, [angle])``
              offset by roi upper left and scaled by roi width and height,
            * list of attributes values with confidences
              ``(attr_name, value, confidence)``
        """
        output = np.transpose(output_layers[0])

        ret_empty = np.float32([]), []

        confidences = output[:, 4]
        keep = confidences > self.confidence_threshold
        if not keep.any():
            return ret_empty

        output = output[keep]

        bboxes = output[:, :4]
        confidences = output[:, 4]
        keep = nms_cpu(
            bboxes,
            confidences,
            self.nms_iou_threshold,
            self.top_k,
        )
        if not keep.any():
            return ret_empty

        output = output[keep]

        # person class id = 0
        class_ids = np.zeros((output.shape[0], 1), dtype=np.float32)
        confidences = output[:, 4:5]
        bboxes = output[:, :4]
        key_points = output[:, 5:]
        key_points = key_points.reshape(key_points.shape[0], -1, 3)

        # scale
        roi_left, roi_top, roi_width, roi_height = roi
        if model.input.maintain_aspect_ratio:
            ratio_x = ratio_y = min(
                model.input.width / roi_width,
                model.input.height / roi_height,
            )
        else:
            ratio_x = model.input.width / roi_width
            ratio_y = model.input.height / roi_height

        bboxes /= np.float32([ratio_x, ratio_y, ratio_x, ratio_y])
        bboxes[:, 0] += roi_left
        bboxes[:, 1] += roi_top

        key_points /= np.float32([ratio_x, ratio_y, 1.0])
        key_points[:, 0] += roi_left
        key_points[:, 1] += roi_top

        bboxes = np.concatenate((class_ids, confidences, bboxes), axis=1)

        attr_name = model.output.attributes[0].name
        key_points = [[(attr_name, pts, 1.0)] for pts in key_points]

        return bboxes, key_points
