"""YOLOv11s detector postprocessing (converter)."""

from typing import Optional, Tuple

import numpy as np

from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ObjectModel
from savant.utils.nms import nms_cpu


class TensorToBBoxConverter(BaseObjectModelOutputConverter):
    """YOLOv11s output to bbox converter."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.0,
        top_k: int = 3000,
        class_ids: Tuple[int] = None,
    ):
        """
        :param confidence_threshold: Select detections with confidence
            greater than specified.
        :param nms_iou_threshold: Class agnostic NMS IoU threshold.
        :param top_k: Maximum number of output detections.
        :param class_ids: Filter detections by class.
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k
        self.class_ids = class_ids
        super().__init__()

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Convert YOLOv11s output [1, 84, 8400] -> bboxes.

        :param output_layers: One output tensor [1, 84, N] or [N, 84].
        :param model: Model definition (needs input shape, aspect/padding flags).
        :param roi: [top, left, width, height] of ROI.
        :return: (class_id, confidence, xc, yc, w, h).
        """
        assert len(output_layers) == 1, "YOLOv11s converter expects single output"
        output = output_layers[0]

        # transpose [84, N] -> [N, 84]
        output = output.T

        bboxes = output[:, :4] # xc, yc, w, h
        scores = output[:, 4:] # class probabilities
        confidences = scores.max(axis=-1)
        class_ids = scores.argmax(axis=-1)

        # filter by class
        if self.class_ids:
            class_mask = np.isin(class_ids, self.class_ids)
            bboxes = bboxes[class_mask]
            class_ids = class_ids[class_mask]
            confidences = confidences[class_mask]

        # filter by confidence
        if self.confidence_threshold:
            conf_mask = confidences > self.confidence_threshold
            bboxes = bboxes[conf_mask]
            class_ids = class_ids[conf_mask]
            confidences = confidences[conf_mask]

        # TODO: ability to filter by size (width, height) and aspect ratio
        # apply class agnostic NMS (all classes are treated as one)
        if self.nms_iou_threshold > 0 and len(confidences) > 1:
            nms_mask = nms_cpu(bboxes, confidences, self.nms_iou_threshold, self.top_k)
            bboxes = bboxes[nms_mask]
            class_ids = class_ids[nms_mask]
            confidences = confidences[nms_mask]

        # select top k
        elif len(confidences) > self.top_k:
            top_k_mask = np.argpartition(confidences, -self.top_k)[-self.top_k :]
            bboxes = bboxes[top_k_mask]
            class_ids = class_ids[top_k_mask]
            confidences = confidences[top_k_mask]

        roi_left, roi_top, roi_width, roi_height = roi

        # scale back to ROI
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


        # correct xc, yc by ROI offset
        bboxes[:, 0] += roi_left
        bboxes[:, 1] += roi_top

        return np.concatenate(
            [
                class_ids.reshape(-1, 1),
                confidences.reshape(-1, 1),
                bboxes,
            ],
            axis=1,
        )
