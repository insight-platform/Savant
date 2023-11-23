import numpy as np
import pytest
import torch
import cupy as cp
from savant.utils.memory_repr import opencv_gpu_mat_as_cupy, cupy_as_opencv_gpu_mat
from savant.utils.memory_repr_pytorch import opencv_gpu_mat_as_pytorch, pytorch_tensor_as_opencv_gpu_mat
import cv2

TORCH_TYPE = [torch.int8, torch.uint8, torch.float32]
NUMPY_TYPE = [np.int8, np.uint8, np.float32]
CUPY_TYPE = [cp.int8, cp.uint8, cp.float32]


def get_opencv_image(input_type, channels=3):

    if channels == 3:
        image_np = np.random.randint(0, 255, (10, 20, 3)).astype(input_type)
    elif channels == 1:
        image_np = np.random.randint(0, 255, (10, 20)).astype(input_type)
    else:
        raise ValueError(f"Unsupported number of channels {channels}")
    image_opencv = cv2.cuda_GpuMat()
    image_opencv.upload(image_np)
    return image_opencv


class TestAsOpenCV:
    @pytest.mark.parametrize("input_type", TORCH_TYPE)
    def test_pytorch(
        self, input_type
    ):
        """Test for pytorch tensors with channels first memory format"""
        pytorch_tensor = torch.randint(0, 255, size=(3, 10, 20), device='cuda').to(input_type)

        img_opencv = pytorch_tensor_as_opencv_gpu_mat(pytorch_tensor)
        np.testing.assert_almost_equal(
            img_opencv.download(),
            pytorch_tensor.permute(1, 2, 0).cpu().numpy(),
        )

    @pytest.mark.parametrize("input_type", TORCH_TYPE)
    def test_pytorch_grayscale(self, input_type):
        """Test for pytorch tensors with grayscale image"""

        # shape - [height, width]
        img_pytorch = torch.randint(0, 255, (10, 20), device='cuda').to(input_type)

        img_opencv = pytorch_tensor_as_opencv_gpu_mat(img_pytorch)
        np.testing.assert_almost_equal(
            img_opencv.download(),
            img_pytorch.cpu().numpy(),
        )

    @pytest.mark.parametrize("input_type", CUPY_TYPE)
    @pytest.mark.parametrize("target_device", ['cuda', 'cpu', None])
    def test_cupy_first_channel_memory(self, input_type, target_device):
        """Test for cupy tensors with channels first memory format"""

        cupy_array = cp.random.randint(0, 255, (10, 20, 3)).astype(input_type)

        opencv_mat = cupy_as_opencv_gpu_mat(cupy_array)
        np.testing.assert_almost_equal(
            opencv_mat.download(),
            cupy_array.get(),
        )

    @pytest.mark.parametrize("input_type", CUPY_TYPE)
    def test_cupy_grayscale(self, input_type):
        """Test for pytorch tensors with grayscale image"""

        cupy_array = cp.random.randint(0, 255, (10, 20)).astype(input_type)

        opencv_gpu_mat = cupy_as_opencv_gpu_mat(cupy_array)
        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            cupy_array.get(),
        )


class TestToTorch:
    @pytest.mark.parametrize("input_type", NUMPY_TYPE)
    def test_opencv(self, input_type):
        opencv_gpu_mat = cv2.cuda_GpuMat()
        opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20, 3)).astype(input_type))

        torch_tensor = opencv_gpu_mat_as_pytorch(opencv_gpu_mat)

        np.testing.assert_almost_equal(
            np.transpose(opencv_gpu_mat.download(), (2, 0, 1)),
            torch_tensor.cpu().numpy()
        )

    @pytest.mark.parametrize("input_type", NUMPY_TYPE)
    def test_opencv_grayscale(self, input_type):
        opencv_gpu_mat = cv2.cuda_GpuMat()
        opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20)).astype(input_type))

        torch_tensor = opencv_gpu_mat_as_pytorch(opencv_gpu_mat)

        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            torch_tensor.cpu().numpy(),
        )


class TestToCUPY:
    @pytest.mark.parametrize("input_type", NUMPY_TYPE)
    def test_opencv(self, input_type):
        opencv_gpu_mat = cv2.cuda_GpuMat()
        opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20, 3)).astype(input_type))

        cupy_array = opencv_gpu_mat_as_cupy(opencv_gpu_mat)

        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            cupy_array.get(),
        )

    @pytest.mark.parametrize("input_type", NUMPY_TYPE)
    def test_opencv_grayscale(self, input_type):
        opencv_gpu_mat = cv2.cuda_GpuMat()
        opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20)).astype(input_type))

        cupy_array = opencv_gpu_mat_as_cupy(opencv_gpu_mat)

        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            cupy_array.get(),
        )
