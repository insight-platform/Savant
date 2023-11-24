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


class TestAsOpenCV:
    @pytest.mark.parametrize("input_type", TORCH_TYPE)
    @pytest.mark.parametrize("channels", [1, 3, 4])
    @pytest.mark.parametrize("memory_format", ['channels_first', 'channels_last'])
    def test_pytorch_3d(
        self, input_type, channels, memory_format
    ):
        """Test for pytorch 3d tensors emulate color image"""
        if memory_format == 'channels_first':
            # shape - [channels, height, width]
            pytorch_tensor = torch.randint(0, 255, size=(channels, 10, 20), device='cuda').to(input_type)
        elif memory_format == 'channels_last':
            # shape - [height, width, channels]
            pytorch_tensor = torch.randint(0, 255, size=(10, 20, channels), device='cuda').to(input_type).permute(2, 0, 1)
        else:
            raise ValueError(f"Unsupported memory format {memory_format}")

        opencv_gpu_mat = pytorch_tensor_as_opencv_gpu_mat(pytorch_tensor)
        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            pytorch_tensor.squeeze(0).cpu().numpy() if channels == 1 else pytorch_tensor.permute(1, 2, 0).cpu().numpy()
        )

        if memory_format == 'channels_first' and channels != 1:
            assert opencv_gpu_mat.cudaPtr() != pytorch_tensor.data_ptr()
        else:
            assert opencv_gpu_mat.cudaPtr() == pytorch_tensor.data_ptr()

    @pytest.mark.parametrize("input_type", TORCH_TYPE)
    def test_pytorch_2d(self, input_type):
        """Test for pytorch tensors  with grayscale image"""

        # shape - [height, width]
        pytorch_tensor = torch.randint(0, 255, (10, 20), device='cuda').to(input_type)

        opencv_gpu_mat = pytorch_tensor_as_opencv_gpu_mat(pytorch_tensor)
        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            pytorch_tensor.cpu().numpy(),
        )

        assert opencv_gpu_mat.cudaPtr() == pytorch_tensor.data_ptr()

    @pytest.mark.parametrize("input_type", CUPY_TYPE)
    @pytest.mark.parametrize("channels", [1, 3, 4])
    @pytest.mark.parametrize("memory_format", ['channels_first', 'channels_last'])
    def test_cupy_3d(self, input_type, channels, memory_format):
        """Test for cupy tensors"""

        if memory_format == 'channels_last':
            cupy_array = cp.random.randint(0, 255, (10, 20, channels)).astype(input_type)
        elif memory_format == 'channels_first':
            cupy_array = cp.random.randint(0, 255, (channels, 10, 20)).astype(input_type).transpose(1, 2, 0)
        else:
            raise ValueError(f"Unsupported memory format {memory_format}")

        opencv_mat = cupy_as_opencv_gpu_mat(cupy_array)
        np.testing.assert_almost_equal(
            opencv_mat.download(),
            cupy_array.squeeze(2).get() if channels == 1 else cupy_array.get(),
        )

        if memory_format == 'channels_first' and channels != 1:
            assert opencv_mat.cudaPtr() != cupy_array.data.ptr
        else:
            assert opencv_mat.cudaPtr() == cupy_array.data.ptr

    @pytest.mark.parametrize("input_type", CUPY_TYPE)
    def test_cupy_2d(self, input_type):
        """Test for pytorch tensors with grayscale image"""

        cupy_array = cp.random.randint(0, 255, (10, 20)).astype(input_type)

        opencv_gpu_mat = cupy_as_opencv_gpu_mat(cupy_array)
        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            cupy_array.get(),
        )

        assert opencv_gpu_mat.cudaPtr() == cupy_array.data.ptr


class TestToTorch:
    @pytest.mark.parametrize("input_type", NUMPY_TYPE)
    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_opencv(self, input_type, channels):
        opencv_gpu_mat = cv2.cuda_GpuMat()
        opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20, channels)).astype(input_type))

        torch_tensor = opencv_gpu_mat_as_pytorch(opencv_gpu_mat)

        np.testing.assert_almost_equal(
            opencv_gpu_mat.download() if channels == 1 else opencv_gpu_mat.download().transpose(2, 0, 1),
            torch_tensor.cpu().numpy()
        )

        assert opencv_gpu_mat.cudaPtr() == torch_tensor.data_ptr()

    @pytest.mark.parametrize("input_type", NUMPY_TYPE)
    def test_opencv_grayscale(self, input_type):
        opencv_gpu_mat = cv2.cuda_GpuMat()
        opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20)).astype(input_type))

        torch_tensor = opencv_gpu_mat_as_pytorch(opencv_gpu_mat)

        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            torch_tensor.cpu().numpy(),
        )

        assert opencv_gpu_mat.cudaPtr() == torch_tensor.data_ptr()


class TestToCUPY:
    @pytest.mark.parametrize("input_type", NUMPY_TYPE)
    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_opencv(self, input_type, channels):
        opencv_gpu_mat = cv2.cuda_GpuMat()
        opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20, channels)).astype(input_type))

        cupy_array = opencv_gpu_mat_as_cupy(opencv_gpu_mat)

        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            cupy_array.get(),
        )

        assert opencv_gpu_mat.cudaPtr() == cupy_array.data.ptr

    @pytest.mark.parametrize("input_type", NUMPY_TYPE)
    def test_opencv_grayscale(self, input_type):
        opencv_gpu_mat = cv2.cuda_GpuMat()
        opencv_gpu_mat.upload(np.random.randint(0, 255, (10, 20)).astype(input_type))

        cupy_array = opencv_gpu_mat_as_cupy(opencv_gpu_mat)

        np.testing.assert_almost_equal(
            opencv_gpu_mat.download(),
            cupy_array.get(),
        )

        assert opencv_gpu_mat.cudaPtr() == cupy_array.data.ptr