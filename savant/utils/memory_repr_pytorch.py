import cv2
import torch

from savant.utils.memory_repr import OpenCVGpuMatCudaArrayInterface, \
    _numpy_type_to_opencv_mat_type


__all__ = [
    'opencv_gpu_mat_as_pytorch',
    'pytorch_tensor_as_opencv_gpu_mat',
]


def opencv_gpu_mat_as_pytorch(gpu_mat: cv2.cuda.GpuMat) -> torch.Tensor:
    """Returns PyTorch tensor for specified OpenCV GpuMat."""
    torch_tensor = torch.tensor(OpenCVGpuMatCudaArrayInterface(gpu_mat), device="cuda")
    if torch_tensor.dim() == 2:
        return torch_tensor
    return torch_tensor.permute(2, 0, 1)


def pytorch_tensor_as_opencv_gpu_mat(tensor: torch.Tensor) -> cv2.cuda_GpuMat:
    """Returns OpenCV GpuMat for specified PyTorch tensor.
    Supports 2 and 3 dims arrays in CHW format for shape and `channel_first`
    memory format only.
    """
    if tensor.dim() not in (2, 3):
        raise ValueError('PyTorch tensor must have 2 or 3 dimensions.')
    if tensor.dim() == 2:
        channels = 1
        tensor_shape = tensor.shape[::-1]
    else:
        channels = tensor.shape[0]
        tensor_shape = tensor.shape[3:0:-1]
        tensor = tensor.permute(1, 2, 0).ravel()
    return cv2.cuda.createGpuMatFromCudaMemory(
        tensor_shape,
        _numpy_type_to_opencv_mat_type(tensor.__cuda_array_interface__['typestr'], channels),
        tensor.__cuda_array_interface__['data'][0],
    )
