import cv2
import torch

from savant.utils.memory_repr import (
    OpenCVGpuMatCudaArrayInterface,
    cuda_array_as_opencv_gpu_mat,
)


def opencv_gpu_mat_as_pytorch_tensor(gpu_mat: cv2.cuda.GpuMat) -> torch.Tensor:
    """Returns PyTorch tensor in HWC format for the given OpenCV GpuMat (zero-copy)."""
    return torch.as_tensor(OpenCVGpuMatCudaArrayInterface(gpu_mat), device='cuda')


def pytorch_tensor_as_opencv_gpu_mat(tensor: torch.Tensor) -> cv2.cuda_GpuMat:
    """Returns OpenCV GpuMat for the given PyTorch tensor (zero-copy).
    The tensor must have 2 or 3 dims in HWC format and C-contiguous layout.

    Use `Tensor.size()` and `Tensor.is_contiguous()` to check if a tensor
    has supported shape format and is contiguous in memory.

    Use `Tensor.transpose()` and `Tensor.contiguous()` to transform a tensor
    if necessary (creates a copy of the tensor).
    """
    return cuda_array_as_opencv_gpu_mat(tensor)
