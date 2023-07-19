"""YOLOv8-seg postprocessing (converter).
TODO: Add `symmetric-padding` support.
"""
from typing import Any, List, Tuple
import numpy as np
import cv2

from savant.base.converter import BaseComplexModelOutputConverter
from savant.deepstream.nvinfer.model import NvInferInstanceSegmentation
from savant.selector.detector import nms_cpu


class TensorToBBoxSegConverter(BaseComplexModelOutputConverter):
    """YOLOv8-seg output converter.

    :param confidence_threshold: confidence threshold (pre-cluster-threshold)
    :param nms_iou_threshold: nms iou threshold
    :param top_k: leave no more than top K bboxes with maximum confidence
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.45,
        top_k: int = 300,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k
        super().__init__()

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: NvInferInstanceSegmentation,
        roi: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, List[List[Tuple[str, Any, float]]]]:
        """Converts model output layer tensors to bbox/seg tensors.

        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: a combination of :py:class:`.BaseObjectModelOutputConverter` and
            corresponding segmentation masks
        """
        # bbox postprocessing
        # shape(4 + num_classes + num_masks, num_boxes) => shape(num_boxes, ...)
        output = output_layers[0].transpose(-1, -2)

        bboxes = output[:, :4]  # xc, yc, width, height
        scores = output[:, 4 : 4 + model.output.num_detected_classes]
        class_ids = np.argmax(scores, axis=-1)
        confidences = np.max(scores, axis=-1)
        masks = output[:, 4 + model.output.num_detected_classes :]

        # filter by confidence
        if self.confidence_threshold and bboxes.shape[0] > 0:
            conf_mask = confidences > self.confidence_threshold
            bboxes = bboxes[conf_mask]
            class_ids = class_ids[conf_mask]
            confidences = confidences[conf_mask]
            masks = masks[conf_mask]

        # select top k
        if bboxes.shape[0] > self.top_k:
            top_k_mask = np.argpartition(confidences, -self.top_k)[-self.top_k :]
            bboxes = bboxes[top_k_mask]
            class_ids = class_ids[top_k_mask]
            confidences = confidences[top_k_mask]
            masks = masks[top_k_mask]

        # nms, class agnostic (all classes are treated as one)
        if self.nms_iou_threshold and bboxes.shape[0] > 0:
            nms_mask = nms_cpu(bboxes, confidences, self.nms_iou_threshold) == 1
            bboxes = bboxes[nms_mask]
            class_ids = class_ids[nms_mask]
            confidences = confidences[nms_mask]
            masks = masks[nms_mask]

        if bboxes.shape[0] == 0:
            return np.float32([]), []

        roi_left, roi_top, roi_width, roi_height = roi

        # scale bboxes
        if model.input.maintain_aspect_ratio:
            ratio_x = ratio_y = max(
                roi_width / model.input.width,
                roi_height / model.input.height,
            )
        else:
            ratio_x = roi_width / model.input.width
            ratio_y = roi_height / model.input.height

        bboxes[:, [0, 2]] *= ratio_x
        bboxes[:, [1, 3]] *= ratio_y

        # round and clip bboxes to cut masks
        # xc, yc, width, height -> left, top, right, bottom
        ltrb_bboxes = np.empty(bboxes.shape, dtype=np.uint16)
        ltrb_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        ltrb_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
        ltrb_bboxes[:, 2] = ltrb_bboxes[:, 0] + bboxes[:, 2]
        ltrb_bboxes[:, 3] = ltrb_bboxes[:, 1] + bboxes[:, 3]
        ltrb_bboxes[:, [0, 2]] = np.clip(ltrb_bboxes[:, [0, 2]], 0, roi_width)
        ltrb_bboxes[:, [1, 3]] = np.clip(ltrb_bboxes[:, [1, 3]], 0, roi_height)

        # correct bboxes
        bboxes[:, 2] = ltrb_bboxes[:, 2] - ltrb_bboxes[:, 0]
        bboxes[:, 3] = ltrb_bboxes[:, 3] - ltrb_bboxes[:, 1]
        bboxes[:, 0] = ltrb_bboxes[:, 0] + bboxes[:, 2] / 2
        bboxes[:, 1] = ltrb_bboxes[:, 1] + bboxes[:, 3] / 2
        # xc, yc to roi left/top
        bboxes[:, 0] += roi_left
        bboxes[:, 1] += roi_top

        # proto masks postprocessing
        # shape(mask_dim, mask_height, mask_width)
        protos = output_layers[1]
        mask_d, mask_h, mask_w = protos.shape
        masks = masks @ protos.reshape(mask_d, -1)
        masks = 1.0 / (1.0 + np.exp(-masks))  # sigmoid
        masks = masks.reshape(-1, mask_h, mask_w)

        # scale masks (transpose to use cv2.resize)
        masks = masks.transpose((1, 2, 0))
        masks = cv2.resize(
            masks,
            (int(ratio_x * model.input.width), int(ratio_y * model.input.height)),
            cv2.INTER_LINEAR,
        )
        masks = (
            masks.transpose((2, 0, 1)) if len(masks.shape) == 3 else masks[None, ...]
        )

        masks = masks > 0.5

        # crop masks
        mask_list = [
            [
                (
                    model.output.attributes[0].name,
                    masks[
                        i,
                        ltrb_bboxes[i, 1] : ltrb_bboxes[i, 3],
                        ltrb_bboxes[i, 0] : ltrb_bboxes[i, 2],
                    ],
                    1.0,
                )
            ]
            for i in range(len(masks))
        ]

        return (
            np.concatenate(
                (
                    class_ids.reshape(-1, 1).astype(np.float32),
                    confidences.reshape(-1, 1),
                    bboxes,
                ),
                axis=1,
            ),
            mask_list,
        )
