"""YOLO base detector postprocessing (converter)."""

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ObjectModel
from savant.utils.nms import nms_cpu


class TensorToBBoxConverter(BaseObjectModelOutputConverter):
    """YOLO detector output to bbox converter."""

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
        """Converts detector output layer tensor to bbox tensor.

        Converter is suitable for PyTorch YOLOv4/v5/v6/v7/v8/v11 models.
        `output_layers` is assumed to consist of
        1) 1 tensor of shape (num_detected_classes+4)xN, or
        2) 1 tensor of shape Nx(num_detected_classes+4+1), or
        3) 2 tensors of shapes Nx1x4 and Nx(num_detected_classes), or
        4) 3 tensors of shapes Nx4, Nx(num_detected_classes), and Nx1, or
        5) 4 tensors (after NMS) of shapes: 1, Nx4, N, N.

        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [left, top, width, height] of the rectangle
            on which the model infers
        :return: BBox tensor (class_id, confidence, xc, yc, width, height, [angle])
            offset by roi upper left and scaled by roi width and height
        """

        if len(output_layers) == 1:
            output = output_layers[0]
            assert model.output.num_detected_classes is not None
            if output.shape[0] == model.output.num_detected_classes + 4:
                output = np.transpose(output)
                scores = output[:, 4:]
            else:
                scores = output[:, 5:] * output[:, 4:5]  # obj_conf * cls_conf
            bboxes = output[:, :4]  # xc, yc, width, height
            class_ids = np.argmax(scores, axis=-1)
            confidences = np.max(scores, axis=-1)

        # YOLOv4
        elif len(output_layers) == 2:
            boxes, scores = output_layers
            # [num, 1, 4] -> [num, 4]
            bboxes = np.squeeze(boxes)
            # YOLOv4 returns [left, top, right, bottom] in normalized coordinates
            bboxes[:, 2] -= bboxes[:, 0]  # width = right - left
            bboxes[:, 3] -= bboxes[:, 1]  # height = bottom - top
            bboxes[:, 0] += bboxes[:, 2] / 2  # convert to xc
            bboxes[:, 1] += bboxes[:, 3] / 2  # convert to yc
            bboxes[:, [0, 2]] *= model.input.width
            bboxes[:, [1, 3]] *= model.input.height
            class_ids = np.argmax(scores, axis=-1)
            confidences = np.max(scores, axis=-1)

        elif len(output_layers) == 3:
            bboxes, scores, class_ids = output_layers
            confidences = np.max(scores, axis=-1)

        elif len(output_layers) == 4:
            num_dets, det_boxes, det_scores, det_classes = output_layers
            num = int(num_dets[0])
            bboxes = det_boxes[:num]
            confidences = det_scores[:num]
            class_ids = det_classes[:num]

            # [0..1] -> model.input
            bboxes[:, [0, 2]] *= model.input.width
            bboxes[:, [1, 3]] *= model.input.height

            # (left, top, right, bottom) -> (xc, yc, width, height)
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]
            bboxes[:, 0] += bboxes[:, 2] / 2
            bboxes[:, 1] += bboxes[:, 3] / 2

        else:
            raise ValueError(
                f'Unsupported number of output layers: {len(output_layers)}'
            )

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

        # transform output coordinates to ROI coordinates
        (scale_x, scale_y), (pad_x, pad_y) = compute_scale_and_pad(
            roi,
            model.input.width,
            model.input.height,
            model.input.maintain_aspect_ratio,
            model.input.symmetric_padding,
        )
        bboxes[:, [0, 2]] *= scale_x + pad_x
        bboxes[:, [1, 3]] *= scale_y + pad_y

        return np.concatenate(
            (
                class_ids.reshape(-1, 1).astype(np.float32),
                confidences.reshape(-1, 1),
                bboxes,
            ),
            axis=1,
        )


@lru_cache()
def compute_scale_and_pad(
    roi: Tuple[float, float, float, float],
    model_input_width: int,
    model_input_height: int,
    maintain_aspect_ratio: bool,
    symmetric_padding: bool,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Get scale parameters for output coordinates.

    :param roi: [left, top, width, height] of the rectangle
        on which the model infers.
    :param model_input_width: Model input width.
    :param model_input_height: Model input height.
    :param maintain_aspect_ratio: If True, scale is computed to maintain aspect ratio.
    :param symmetric_padding: If True, padding is applied symmetrically.
    :return: Scale parameters ((scale_x, scale_y), (pad_x, pad_y)).
    """
    roi_left, roi_top, roi_width, roi_height = roi

    pad_x, pad_y = roi_left, roi_top

    if maintain_aspect_ratio:
        scale_x = scale_y = max(
            roi_width / model_input_width,
            roi_height / model_input_height,
        )
        if symmetric_padding:
            pad_x += ((model_input_width - roi_width / scale_x) / 2) * scale_x
            pad_y += ((model_input_height - roi_height / scale_y) / 2) * scale_y
    else:
        scale_x = roi_width / model_input_width
        scale_y = roi_height / model_input_height

    return (scale_x, scale_y), (pad_x, pad_y)
