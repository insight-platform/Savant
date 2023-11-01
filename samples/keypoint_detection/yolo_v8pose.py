"""YOLOv8pose postprocessing (converter).
Postprocessing for YOLOv8pose detector model
https://github.com/marcoslucianops/DeepStream-Yolo-Pose/blob/master/docs/YOLOv8_Pose.md
based on code from https://github.com/marcoslucianops/DeepStream-Yolo-Pose>
"""
from typing import Any, Tuple

import numpy as np

from numba.typed import List
from savant.base.converter import BaseComplexModelOutputConverter
from savant.base.model import ComplexModel
from savant.utils.nms import nms_cpu


class YoloV8faceConverter(BaseComplexModelOutputConverter):
    """`YOLOv8pose <https://github.com/marcoslucianops/DeepStream-Yolo-Pose>`_ output
    tensor to bbox and key-point converter."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        nms_iou_threshold: float = 0.45,
        **kwargs,
    ):
        """Initialize YOLOv8pose converter."""
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ComplexModel,
        roi: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, List[List[Tuple[str, Any, float]]]]:
        """Converts output layer tensor to bbox tensor and addition
        attribute(key points).

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
        bboxes = np.float32(output_layers[0])
        scores = np.float32(output_layers[1])
        kpts = np.float32(output_layers[2])

        bboxes = bboxes[scores[:, 0] > self.confidence_threshold]
        kpts = kpts[scores[:, 0] > self.confidence_threshold]
        scores = scores[scores[:, 0] > self.confidence_threshold]

        atr_name = model.output.attributes[0].name

        # changing the scale of the model output according to
        # the resolution of the video stream
        # models works with aspect ratio saving
        ratio = max(
            roi[3] / model.input.shape[1],
            roi[2] / model.input.shape[2],
        )
        pad_x = (model.input.shape[2] - roi[2] / ratio) / 2.0
        pad_y = (model.input.shape[1] - roi[3] / ratio) / 2.0

        if bboxes is not None and bboxes.size:
            keep = nms_cpu(
                bboxes,
                scores[:, 0],
                self.nms_iou_threshold,
            )
            bboxes = bboxes[keep == 1]
            scores = scores[keep == 1]
            kpts = kpts[keep == 1].reshape(-1, 17, 3)
            mean_conf = kpts[:, :, 2].mean(1)
            kpts = kpts[:, :, :2]
            bboxes -= np.tile(np.float32([pad_x, pad_y]), 2)
            bboxes *= np.tile(np.float32([ratio, ratio]), 2)
            kpts -= np.float32([pad_x, pad_y])
            kpts *= np.float32([ratio, ratio])
            bboxes = np.concatenate(
                (np.zeros((bboxes.shape[0], 1), dtype=np.float32), scores, bboxes),
                axis=1,
            )
            key_points = [
                [(atr_name, lms, conf)]
                for lms, conf in zip(kpts.reshape(-1, 34).tolist(), mean_conf.tolist())
            ]
            return bboxes, key_points

        return np.float32([]), []
