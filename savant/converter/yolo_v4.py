"""YOLOv4 detector postprocessing (converter).

Based on code from https://github.com/Tianxiaomo/pytorch-YOLOv4
"""

from typing import Tuple

import numpy as np

from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ObjectModel


class TensorToBBoxConverter(BaseObjectModelOutputConverter):
    """`YOLOv4 <https://github.com/Tianxiaomo/pytorch-YOLOv4>`_ output to bbox
    converter."""

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Converts detector output layer tensor to bbox tensor.

        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: BBox tensor (class_id, confidence, xc, yc, width, height, [angle])
            offset by roi upper left and scaled by roi width and height
        """
        boxes, confs = output_layers
        roi_left, roi_top, roi_width, roi_height = roi

        # [num, 1, 4] -> [num, 4]
        bboxes = np.squeeze(boxes)

        # TODO: Return to original YOLOv4 xc, yc
        # left, top, right, bottom => xc, yc, width, height
        bboxes[:, 2] -= bboxes[:, 0]
        bboxes[:, 3] -= bboxes[:, 1]
        bboxes[:, 0] += bboxes[:, 2] / 2
        bboxes[:, 1] += bboxes[:, 3] / 2

        # scale
        if model.input.maintain_aspect_ratio:
            bboxes *= min(roi_width, roi_height)
        else:
            bboxes[:, [0, 2]] *= roi_width
            bboxes[:, [1, 3]] *= roi_height
        # correct xc, yc
        bboxes[:, 0] += roi_left
        bboxes[:, 1] += roi_top

        # [num, num_classes] --> [num]
        confidences = np.max(confs, axis=-1)
        class_ids = np.argmax(confs, axis=-1)

        return np.concatenate(
            (
                class_ids.reshape(-1, 1).astype(np.float32),
                confidences.reshape(-1, 1),
                bboxes,
            ),
            axis=1,
        )
