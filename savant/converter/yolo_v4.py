"""YOLOv4 detector postprocessing (converter).

Based on code from https://github.com/Tianxiaomo/pytorch-YOLOv4
"""

from typing import Optional, Tuple

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
    ) -> Optional[np.ndarray]:
        """Converts detector output layer tensor to bbox tensor.

        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio, symmetric_padding
        :param roi: [left, top, width, height] of the rectangle
            on which the model infers
        :return: BBox tensor (class_id, confidence, xc, yc, width, height, [angle])
            offset by roi upper left and scaled by roi width and height
        """
        boxes, confs = output_layers
        roi_left, roi_top, roi_width, roi_height = roi

        # [num, 1, 4] -> [num, 4]
        bboxes = np.squeeze(boxes)

        # YOLOv4 returns [left, top, right, bottom] in normalized coordinates
        bboxes[:, 2] -= bboxes[:, 0]  # width = right - left
        bboxes[:, 3] -= bboxes[:, 1]  # height = bottom - top
        bboxes[:, 0] += bboxes[:, 2] / 2  # convert to xc
        bboxes[:, 1] += bboxes[:, 3] / 2  # convert to yc

        input_w = model.input.width
        input_h = model.input.height

        # Convert normalized coordinates to input space (pixels)
        bboxes[:, 0] *= input_w
        bboxes[:, 1] *= input_h
        bboxes[:, 2] *= input_w
        bboxes[:, 3] *= input_h

        if model.input.maintain_aspect_ratio:
            scale = min(model.input.width / roi_width,
                        model.input.height / roi_height)
            inv_scale = 1.0 / scale
            bboxes *= inv_scale
        
            if model.input.symmetric_padding:
                new_w = roi_width * scale
                new_h = roi_height * scale
        
                # Convert to ROI coordinates
                pad_x = (model.input.width - new_w) * 0.5 * inv_scale
                pad_y = (model.input.height - new_h) * 0.5 * inv_scale
        
                bboxes[:, 0] -= pad_x
                bboxes[:, 1] -= pad_y
        else:
            bboxes[:, [0, 2]] *= roi_width / model.input.width
            bboxes[:, [1, 3]] *= roi_height / model.input.height

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
