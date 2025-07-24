"""YOLOv8face detector postprocessing (converter).

Based on code from https://github.com/derronqi/yolov8-face>
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from savant.base.converter import BaseComplexModelOutputConverter
from savant.base.model import ComplexModel
from savant.utils.nms import nms_cpu


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
        attr_name = model.output.attributes[0].name

        roi_left, roi_top, roi_width, roi_height = roi
        ratio_width = roi_width / model.input.shape[2]
        ratio_height = roi_height / model.input.shape[1]

        raw_predictions = np.transpose(output_layers[0])

        selected_predictions = raw_predictions[
            raw_predictions[:, 4] > self.confidence_threshold
        ]
        if selected_predictions.shape[0] == 0:
            return

        keep = nms_cpu(
            selected_predictions[:, :4],
            selected_predictions[:, 4],
            self.nms_iou_threshold,
            selected_predictions.shape[0],
        )

        selected_nms_predictions = selected_predictions[keep]
        if selected_nms_predictions.shape[0] == 0:
            return

        xywh = selected_nms_predictions[:, :4]
        conf = selected_nms_predictions[:, 4:5]
        class_num = np.zeros_like(conf)

        # Scale and shift bounding box coordinates
        if model.input.maintain_aspect_ratio:
            scale = min(
                model.input.width / roi_width,
                model.input.height / roi_height,
            )
            xywh /= scale

            if model.input.symmetric_padding:
                new_width = roi_width * scale
                new_height = roi_height * scale

                # Convert to ROI coordinates
                pad_x = (model.input.width - new_width) / (2 * scale)
                pad_y = (model.input.height - new_height) / (2 * scale)

                xywh[:, 0] -= pad_x  # xc
                xywh[:, 1] -= pad_y  # yc
        else:
            # Without aspect ratio preservation, use direct scaling
            xywh *= np.tile(np.float32([ratio_width, ratio_height]), 2)

        # Offset bounding box centers to full-frame coordinates
        xywh[:, 0] += roi_left  # x center
        xywh[:, 1] += roi_top  # y center

        bbox_output = np.concatenate((class_num, conf, xywh), axis=1)

        # Process landmarks (5 points, each with x, y, conf)
        landmarks = (
            selected_nms_predictions[:, 5:20]
            * np.tile(np.float32([ratio_width, ratio_height, 1.0]), 5)
        ).reshape(-1, 5, 3)
        landmarks[:, :, 0] += roi_left  # x
        landmarks[:, :, 1] += roi_top  # y

        landmarks_output = [
            [(attr_name, lms, conf)]
            for lms, conf in zip(
                landmarks[:, :, :2].reshape(-1, 10).tolist(),
                landmarks[:, :, 2].mean(1),
            )
        ]

        return bbox_output, landmarks_output
