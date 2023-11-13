"""YOLOv8-seg postprocessing (converter)."""
from typing import Any, List, Tuple

import cupy as cp
import cv2
import numpy as np

from savant.base.converter import BaseComplexModelOutputConverter
from savant.deepstream.nvinfer.model import NvInferInstanceSegmentation
from savant.utils.nms import nms_gpu


class TensorToBBoxSegConverter(BaseComplexModelOutputConverter):
    """YOLOv8-seg output converter.

    :param confidence_threshold: confidence threshold (pre-cluster-threshold)
    :param nms_iou_threshold: nms iou threshold
    :param top_k: leave no more than top K bboxes with maximum confidence
    """

    gpu = True

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
        *output_layers: cp.ndarray,
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

        tensors, masks = _postproc(
            output=output_layers[0],
            protos=output_layers[1],
            num_detected_classes=model.output.num_detected_classes,
            nms_iou_threshold=self.nms_iou_threshold,
            confidence_threshold=self.confidence_threshold,
            top_k=self.top_k,
        )

        if tensors.shape[0] == 0:
            return tensors.get(), []

        roi_left, roi_top, roi_width, roi_height = roi

        if model.input.maintain_aspect_ratio:
            ratio_x = ratio_y = max(
                roi_width / model.input.width,
                roi_height / model.input.height,
            )
        else:
            ratio_x = roi_width / model.input.width
            ratio_y = roi_height / model.input.height

        # scale & shift bboxes
        tensors[:, [2, 4]] *= ratio_x
        tensors[:, [3, 5]] *= ratio_y
        tensors[:, 2] += roi_left
        tensors[:, 3] += roi_top

        # scale masks & prepare mask list
        mask_width = int(ratio_x * model.input.width)
        mask_height = int(ratio_y * model.input.height)

        mask_list = []
        for i in range(masks.shape[0]):
            gpu_mat = cv_cuda_gpumat_from_cp_array(masks[i])

            resized_gpu_mat = cv2.cuda.resize(
                src=gpu_mat,
                dsize=(mask_width, mask_height),
                interpolation=cv2.INTER_LINEAR,
            )

            mask = cp_array_from_cv_cuda_gpumat(resized_gpu_mat) > 0.5

            mask_list.append(
                [
                    (
                        model.output.attributes[0].name,
                        mask[
                            max(0, int(tensors[i, 3] - tensors[i, 5] / 2)) : min(
                                mask_height, int(tensors[i, 3] + tensors[i, 5] / 2)
                            ),
                            max(0, int(tensors[i, 2] - tensors[i, 4] / 2)) : min(
                                mask_width, int(tensors[i, 2] + tensors[i, 4] / 2)
                            ),
                        ].get(),
                        1.0,
                    )
                ]
            )

        return tensors.get(), mask_list


def cv_cuda_gpumat_from_cp_array(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    """TODO: Refactor
    https://github.com/rapidsai/cucim/issues/329"""
    assert len(arr.shape) in (2, 3), "CuPy array must have 2 or 3 dimensions to be a valid GpuMat"
    type_map = {
        cp.dtype('uint8'): cv2.CV_8U,
        cp.dtype('int8'): cv2.CV_8S,
        cp.dtype('uint16'): cv2.CV_16U,
        cp.dtype('int16'): cv2.CV_16S,
        cp.dtype('int32'): cv2.CV_32S,
        cp.dtype('float32'): cv2.CV_32F,
        cp.dtype('float64'): cv2.CV_64F,
    }
    depth = type_map.get(arr.dtype)
    assert depth is not None, 'Unsupported CuPy array dtype'
    channels = 1 if len(arr.shape) == 2 else arr.shape[2]
    # equivalent to unexposed opencv C++ macro CV_MAKETYPE(depth,channels):
    # (depth&7) + ((channels - 1) << 3)
    mat_type = depth + ((channels - 1) << 3)
    mat = cv2.cuda.createGpuMatFromCudaMemory(
        arr.__cuda_array_interface__['shape'][1::-1],
        mat_type,
        arr.__cuda_array_interface__['data'][0],
    )
    return mat


def cp_array_from_cv_cuda_gpumat(mat: cv2.cuda.GpuMat) -> cp.ndarray:
    """TODO: Refactor
    https://github.com/rapidsai/cucim/issues/329"""
    class CudaArrayInterface:
        def __init__(self, gpu_mat: cv2.cuda.GpuMat):
            w, h = gpu_mat.size()
            type_map = {
                cv2.CV_8U: '|u1',
                cv2.CV_8S: '|i1',
                cv2.CV_16U: '<u2',
                cv2.CV_16S: '<i2',
                cv2.CV_32S: '<i4',
                cv2.CV_32F: '<f4',
                cv2.CV_64F: '<f8',
            }
            self.__cuda_array_interface__ = {
                'version': 3,
                'shape': (h, w, gpu_mat.channels())
                if gpu_mat.channels() > 1
                else (h, w),
                'typestr': type_map[gpu_mat.depth()],
                'descr': [('', type_map[gpu_mat.depth()])],
                'stream': 1,
                'strides': (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1())
                if gpu_mat.channels() > 1
                else (gpu_mat.step, gpu_mat.elemSize()),
                'data': (gpu_mat.cudaPtr(), False),
            }

    arr = cp.asarray(CudaArrayInterface(mat))

    return arr


@cp.fuse()
def sigmoid(a: cp.ndarray) -> cp.ndarray:
    return cp.divide(1, (1 + cp.exp(-a)))


def _postproc(
    output: cp.ndarray,
    protos: cp.ndarray,
    num_detected_classes: int,
    nms_iou_threshold: float,
    confidence_threshold: float,
    top_k: int,
) -> Tuple[cp.ndarray, cp.ndarray]:
    # bbox postprocessing
    # shape(4 + num_classes + num_masks, num_boxes) => shape(num_boxes, ...)
    output = output.transpose(-1, -2)

    bboxes = output[:, :4]  # xc, yc, width, height
    scores = output[:, 4 : 4 + num_detected_classes]
    class_ids = cp.argmax(scores, axis=-1)
    confidences = cp.max(scores, axis=-1)
    masks = output[:, 4 + num_detected_classes :]

    # filter by confidence
    if confidence_threshold and bboxes.shape[0] > 0:
        conf_mask = confidences > confidence_threshold
        bboxes = bboxes[conf_mask]
        class_ids = class_ids[conf_mask]
        confidences = confidences[conf_mask]
        masks = masks[conf_mask]

    # select top k
    if bboxes.shape[0] > top_k:
        top_k_mask = cp.argpartition(confidences, -top_k)[-top_k:]
        bboxes = bboxes[top_k_mask]
        class_ids = class_ids[top_k_mask]
        confidences = confidences[top_k_mask]
        masks = masks[top_k_mask]

    # nms, class agnostic (all classes are treated as one)
    if nms_iou_threshold and bboxes.shape[0] > 0:
        nms_mask = nms_gpu(bboxes, confidences, nms_iou_threshold) == 1
        bboxes = bboxes[nms_mask]
        class_ids = class_ids[nms_mask]
        confidences = confidences[nms_mask]
        masks = masks[nms_mask]

    if bboxes.shape[0] == 0:
        return cp.empty((0, 0), dtype=cp.float32), cp.empty((0, 0, 0), dtype=cp.float32)

    tensors = cp.empty((bboxes.shape[0], 6), dtype=cp.float32)
    tensors[:, 0] = class_ids
    tensors[:, 1] = confidences
    tensors[:, 2:] = bboxes

    # proto masks postprocessing
    mask_dim, mask_height, mask_width = protos.shape
    masks = sigmoid(cp.ascontiguousarray(masks) @ protos.reshape(mask_dim, -1))
    masks = masks.reshape(-1, mask_height, mask_width)

    return tensors, masks
