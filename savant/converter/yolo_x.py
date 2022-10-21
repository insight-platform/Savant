"""YOLOX detector postprocessing (converter)."""
from typing import Tuple
from functools import lru_cache
import numpy as np
from savant.converter.yolo import TensorToBBoxConverter as YoloTensorToBBoxConverter
from savant.base.model import ObjectModel


class TensorToBBoxConverter(YoloTensorToBBoxConverter):
    """`YOLOX <https://github.com/Megvii-BaseDetection/YOLOX>`_ output to bbox
    converter."""

    def __init__(
        self,
        decode: bool = False,
        confidence_threshold: float = 0.25,
        top_k: int = 3000,
    ):
        """
        :param decode: Decode output before convert.
            We must decode output if we didn't use `--decode_in_inference`
            when exporting to ONNX.
        :param confidence_threshold: Select detections with confidence
            greater than specified.
        :param top_k: Maximum number of output detections.
        """
        self.decode = decode
        super().__init__(confidence_threshold=confidence_threshold, top_k=top_k)

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Converts YOLOX detector output layer tensor to bbox tensor.

        :param output_layers: Output layer tensor
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: BBox tensor (class_id, confidence, xc, yc, width, height, [angle])
            offset by roi upper left and scaled by roi width and height
        """
        output = output_layers[0]

        if self.decode:
            grids, strides = _get_grids_strides(model.input.height, model.input.width)
            output[..., :2] = (output[..., :2] + grids) * strides
            output[..., 2:4] = np.exp(output[..., 2:4]) * strides

        return super().__call__(*[output], model=model, roi=roi)


@lru_cache()
def _get_grids_strides(input_height: int, input_width: int, yolo_p6: bool = False):

    grids = []
    expanded_strides = []

    if not yolo_p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [input_height // stride for stride in strides]
    wsizes = [input_width // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        x_idxs, y_idxs = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((x_idxs, y_idxs), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)

    return grids, expanded_strides
