"""YOLOv5face detector postprocessing (converter).

Based on code from https://github.com/deepcam-cn/yolov5-face>
"""
from typing import Any, Tuple

import numpy as np
from numba.typed import List

from savant.base.converter import BaseComplexModelOutputConverter
from savant.base.model import ComplexModel
from savant.selector.detector import nms_cpu


class YoloV5faceConverter(BaseComplexModelOutputConverter):
    """`YOLOv5face <https://github.com/deepcam-cn/yolov5-face>`_ output to bbox
    and landmarks converter."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        nms_iou_threshold: float = 0.5,
        **kwargs,
    ):
        """Initialize YOLOv5face converter."""
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ComplexModel,
        roi: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, List[List[Tuple[str, Any, float]]]]:
        """Converts detector output layer tensor to bbox tensor and addition
        attribute(landmark).

        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: a combination of :py:class:`.BaseObjectModelOutputConverter` and
            :py:class:`.BaseAttributeModelOutputConverter` outputs:

            * BBox tensor ``(class_id, confidence, xc, yc, width, height, [angle])``
              offset by roi upper left and scaled by roi width and height,
            * list of attributes values with confidences
              ``(attr_name, value, confidence)``
        """
        atr_name = model.output.attributes[0].name
        ration_width = roi[2] / model.input.shape[2]
        ratio_height = roi[3] / model.input.shape[1]
        raw_predictions = np.float32(output_layers[0])
        if raw_predictions.size:
            raw_predictions[:, 4:5] *= raw_predictions[:, 15:16]
            selected_preds = raw_predictions[
                raw_predictions[:, 4] > self.confidence_threshold
            ]
            keep = nms_cpu(
                selected_preds[:, :4],
                selected_preds[:, 4],
                self.nms_iou_threshold,
            )
            selected_nms_prediction = selected_preds[keep == 1]
            xywh = selected_nms_prediction[:, :4]
            conf = selected_nms_prediction[:, 4:5]
            class_num = np.zeros_like(conf, dtype=np.float32)
            xywh *= np.tile(np.float32([ration_width, ratio_height]), 2)
            landmarks = selected_nms_prediction[:, 5:15] * np.tile(
                np.float32([ration_width, ratio_height]), 5
            )
            bbox = np.concatenate((class_num, conf, xywh), axis=1)
            landmarks_output = list(
                map(lambda x: [(atr_name, x, 1.0)], landmarks.tolist())
            )
            return bbox, landmarks_output
        return np.float32([]), []
