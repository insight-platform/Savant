"""YOLO base detector postprocessing (converter).
Supports YOLOv5/v6/v7/v8/v11 ONNX heads:
- one tensor shaped [N, (4 + C)]  (no objectness)
- one tensor shaped [N, (5 + C)]  (with objectness at col 4)
- or transposed [(4 + C), N] / [(5 + C), N]
- OR 3/4 tensors after NMS

Now with proper reverse-letterbox for maintain_aspect_ratio + symmetric_padding.
"""

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
        class_ids: Tuple[int, ...] | None = None,
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

    def _decode_single_head(
        self, output: np.ndarray, num_classes: Optional[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (bboxes_xywh, confidences, class_ids) from a single output tensor.
        Handles both [N, 4+C]/[N, 5+C] and transposed [(4+C),N]/[(5+C),N].
        """
        # Squeeze batch dim if present
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]

        # Try to detect transposed layout using num_classes if provided
        def _is_transposed(arr: np.ndarray) -> bool:
            if num_classes is None:
                # Heuristic: if first dim is "plausible" channel count (84/85/etc.)
                return arr.shape[0] in (84, 85, 92, 93)  # typical for COCO and similar
            return arr.shape[0] in (num_classes + 4, num_classes + 5)

        transposed = _is_transposed(output)
        if transposed:
            output = output.T  # -> [N, D]

        D = output.shape[1]
        # Determine if objectness column exists
        if num_classes is not None:
            has_obj = D == (num_classes + 5)
        else:
            # Heuristic: if D-5 looks like a reasonable class count
            has_obj = D >= 10 and (D - 5) in (80, 1, 2, 5, 6, 7, 8, 10, 20, 91)

        bboxes = output[:, :4].astype(np.float32)  # xc, yc, w, h
        if has_obj:
            # obj * class_prob
            scores_mat = output[:, 5:] * output[:, 4:5]
            confidences = scores_mat.max(axis=-1)
            class_ids = scores_mat.argmax(axis=-1).astype(np.int32)
        else:
            # already class scores (no obj column)
            scores_mat = output[:, 4:]
            confidences = scores_mat.max(axis=-1)
            class_ids = scores_mat.argmax(axis=-1).astype(np.int32)

        return bboxes, confidences.astype(np.float32), class_ids

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """Converts detector output layer tensor to bbox tensor.
    
        This converter handles common YOLO heads (incl. YOLOv11 single-head `output0`)
        in either `[N, 4+…] / [N, 5+…]` or transposed `[(4+…), N] / [(5+…), N]` forms,
        and correctly maps boxes back from model input space to the ROI (reverse-letterbox
        if `maintain_aspect_ratio` with `symmetric_padding` is enabled).
    
        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: BBox tensor (class_id, confidence, xc, yc, width, height, [angle])
            offset by roi upper left and scaled by roi width and height
        """
        assert len(output_layers) in (1, 3, 4)

        if len(output_layers) == 1:
            output = output_layers[0]
            num_classes = getattr(model.output, "num_detected_classes", None)
            bboxes, confidences, class_ids = self._decode_single_head(output, num_classes)

        elif len(output_layers) == 3:
            # Already NMS'ed: (bboxes_xywh, scores_per_class, class_ids)
            bboxes, scores, class_ids = output_layers
            confidences = scores.max(axis=-1).astype(np.float32)

        else:
            # TensorRT NMS style: (num, boxes_ltrb_norm, scores, classes)
            num_dets, det_boxes, det_scores, det_classes = output_layers
            num = int(det_boxes.shape[0]) if det_boxes.ndim == 2 else int(num_dets[0])
            bboxes = det_boxes[:num].astype(np.float32)  # [N,4] LTRB normalized
            confidences = det_scores[:num].astype(np.float32)
            class_ids = det_classes[:num].astype(np.int32)

            # [0..1] -> model.input, then LTRB -> XYWH
            bboxes[:, [0, 2]] *= model.input.width
            bboxes[:, [1, 3]] *= model.input.height
            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]
            bboxes[:, 0] += bboxes[:, 2] / 2
            bboxes[:, 1] += bboxes[:, 3] / 2

        # --- class filter
        if self.class_ids:
            mask = np.isin(class_ids, self.class_ids)
            bboxes, confidences, class_ids = bboxes[mask], confidences[mask], class_ids[mask]

        # --- confidence filter
        if self.confidence_threshold > 0:
            mask = confidences > self.confidence_threshold
            bboxes, confidences, class_ids = bboxes[mask], confidences[mask], class_ids[mask]

        # --- NMS (class-agnostic)
        if self.nms_iou_threshold > 0 and bboxes.shape[0] > 1:
            keep = nms_cpu(bboxes, confidences, self.nms_iou_threshold, self.top_k)
            bboxes, confidences, class_ids = bboxes[keep], confidences[keep], class_ids[keep]
        elif bboxes.shape[0] > self.top_k:
            idx = np.argpartition(confidences, -self.top_k)[-self.top_k :]
            bboxes, confidences, class_ids = bboxes[idx], confidences[idx], class_ids[idx]

        # --- map from model input space -> ROI space
        roi_left, roi_top, roi_w, roi_h = roi

        if model.input.maintain_aspect_ratio:
            # reverse letterbox: remove padding, then de-scale
            scale = min(model.input.width / roi_w, model.input.height / roi_h)
            if scale <= 0:
                return None
            # de-scale w,h,xc,yc
            bboxes /= scale

            if getattr(model.input, "symmetric_padding", False):
                new_w = roi_w * scale
                new_h = roi_h * scale
                pad_x = (model.input.width - new_w) / 2.0
                pad_y = (model.input.height - new_h) / 2.0
                # convert pad from model space to ROI space
                pad_x /= scale
                pad_y /= scale
                bboxes[:, 0] -= pad_x  # xc
                bboxes[:, 1] -= pad_y  # yc
        else:
            # simple affine scaling
            bboxes[:, [0, 2]] /= (model.input.width / roi_w)
            bboxes[:, [1, 3]] /= (model.input.height / roi_h)

        # offset by ROI top-left
        bboxes[:, 0] += roi_left
        bboxes[:, 1] += roi_top

        # --- pack result
        return np.concatenate(
            (
                class_ids.reshape(-1, 1).astype(np.float32),
                confidences.reshape(-1, 1).astype(np.float32),
                bboxes.astype(np.float32),
            ),
            axis=1,
        )
