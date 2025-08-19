"""YOLOv8face detector postprocessing (converter).

Based on code from https://github.com/derronqi/yolov8-face>
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from savant.base.converter import BaseComplexModelOutputConverter
from savant.base.model import ComplexModel
from savant.utils.nms import nms_cpu

from .yolo import compute_scale_and_pad


class YoloV8faceConverter(BaseComplexModelOutputConverter):
    """`YOLOv8face <https://github.com/derronqi/yolov8-face>`_ output to bbox
    and landmarks converter."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        nms_iou_threshold: float = 0.5,
        **kwargs,
    ):
        """Initialize YOLOv8-face converter."""
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ComplexModel,
        roi: Tuple[float, float, float, float],
    ) -> Optional[Tuple[np.ndarray, List[List[Tuple[str, Any, float]]]]]:
        """Converts detector output layer tensor to bbox tensor and additional
        attributes (landmarks).

        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [left, top, width, height] of the rectangle
            on which the model infers
        :return: a combination of :py:class:`.BaseObjectModelOutputConverter` and
            :py:class:`.BaseAttributeModelOutputConverter` outputs:

            * BBox tensor ``(class_id, confidence, xc, yc, width, height, [angle])``
              offset by roi upper left and scaled by roi width and height,
            * list of attributes values with confidences
              ``(attr_name, value, confidence)``
        """
        raw_predictions = np.transpose(output_layers[0])

        selected_predictions = raw_predictions[
            raw_predictions[:, 4] > self.confidence_threshold
        ]
        if selected_predictions.shape[0] == 0:
            return None

        keep = nms_cpu(
            selected_predictions[:, :4],
            selected_predictions[:, 4],
            self.nms_iou_threshold,
            selected_predictions.shape[0],
        )
        selected_nms_predictions = selected_predictions[keep]
        if selected_nms_predictions.shape[0] == 0:
            return None

        bboxes = selected_nms_predictions[:, :4]
        confidences = selected_nms_predictions[:, 4:5]
        class_ids = np.zeros_like(confidences)

        # process landmarks (5 points, each with x, y, conf)
        landmarks = selected_nms_predictions[:, 5:20].reshape(-1, 5, 3)

        # transform output coordinates to ROI coordinates
        (scale_x, scale_y), (pad_x, pad_y) = compute_scale_and_pad(
            roi,
            model.input.width,
            model.input.height,
            model.input.maintain_aspect_ratio,
            model.input.symmetric_padding,
        )
        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y
        bboxes[:, 0] += pad_x
        bboxes[:, 1] += pad_y
        landmarks[:, :, 0] *= scale_x
        landmarks[:, :, 0] += pad_x
        landmarks[:, :, 1] *= scale_y
        landmarks[:, :, 1] += pad_y

        bbox_output = np.concatenate((class_ids, confidences, bboxes), axis=1)

        attr_name = model.output.attributes[0].name
        landmarks_output = [
            [(attr_name, lms, conf)]
            for lms, conf in zip(
                landmarks[:, :, :2].reshape(-1, 10).tolist(),
                landmarks[:, :, 2].mean(1),
            )
        ]

        return bbox_output, landmarks_output
