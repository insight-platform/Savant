import cupy
import numpy as np
import pytest
import torch
import cupy as cp
from savant.utils.memory_repr import as_opencv, as_pytorch
import cv2


class TestToOpencv:

    @pytest.mark.parametrize("input_device", ['cuda', 'cpu'])
    @pytest.mark.parametrize("input_type", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("target_device", ['cuda', 'cpu', None])
    def test_pytorch_channel_first_memory(
            self,
            input_device,
            input_type,
            target_device
    ):
        """ Test for pytorch tensors with channels first memory format
        """
        # cf - channel first
        # cl - channel last

        # shape - [channels, height, width]
        img_cf_shape_cf_memory = torch.randint(0, 255, size=(3, 10, 20))\
            .to(input_type).to(input_device)
        img_opencv = as_opencv(img_cf_shape_cf_memory, device=target_device)
        np.testing.assert_almost_equal(
            img_opencv if isinstance(img_opencv, np.ndarray) else img_opencv.download(),
            img_cf_shape_cf_memory.permute(1, 2, 0).cpu().numpy()
        )

        # shape - [height, width, channels]. memory_format=torch.channels_fisrt
        img_cl_shape_cf_memory = img_cf_shape_cf_memory.permute(1, 2, 0)
        img_opencv = as_opencv(
            img_cl_shape_cf_memory,
            input_format='channel_last',
            device=target_device
        )
        np.testing.assert_almost_equal(
            img_opencv if isinstance(img_opencv, np.ndarray) else img_opencv.download(),
            img_cl_shape_cf_memory.cpu().numpy(),
        )

    @pytest.mark.parametrize("input_device", ['cuda', 'cpu'])
    @pytest.mark.parametrize("input_type", [np.uint8, np.float32])
    @pytest.mark.parametrize("target_device", ['cuda', 'cpu', None])
    def test_pytorch_last_channel_memory(self, input_device, input_type, target_device):
        """ Test for pytorch tensors with channels last memory format which
        is implemented by using numpy.ndarray
        """

        # shape - [height, width, channels]
        np_image = np.random.randint(0, 255, (10, 20, 3)).astype(input_type)
        img_cf_shape_cl_memory = torch.as_tensor(np_image).to(input_device)

        img_opencv = as_opencv(
            img_cf_shape_cl_memory,
            input_format='channel_last',
            device=target_device
        )
        np.testing.assert_almost_equal(
            img_opencv if isinstance(img_opencv, np.ndarray) else img_opencv.download(),
            np_image
        )

        # shape - [channels, height, width]. memory_format=torch.channel_last
        img_cl_shape_cl_memory = img_cf_shape_cl_memory.permute(2, 0, 1)
        img_opencv = as_opencv(
            img_cl_shape_cl_memory,
            device=target_device
        )
        np.testing.assert_almost_equal(
            img_opencv if isinstance(img_opencv, np.ndarray) else img_opencv.download(),
            np_image
        )

    @pytest.mark.parametrize("input_type", [torch.uint8, torch.float32])
    @pytest.mark.parametrize("target_device", ['cuda', 'cpu', None])
    def test_cupy_first_channel_memory(self, input_type, target_device):
        """ Test for cupy tensors with channels first memory format
        """
        # shape - [channels, height, width]
        img_cf_shape_cf_memory = torch.randint(0, 255, size=(3, 10, 20)) \
            .to(input_type).to('cuda')
        img_cf_shape_cf_memory = cupy.asarray(img_cf_shape_cf_memory)

        img_opencv = as_opencv(
            img_cf_shape_cf_memory,
            input_format='channel_first',
            device=target_device
        )
        np.testing.assert_almost_equal(
            img_opencv if isinstance(img_opencv, np.ndarray) else img_opencv.download(),
            cp.transpose(img_cf_shape_cf_memory, (1, 2, 0)).get()
        )

        # shape - [height, width, channels]
        img_cl_shape_cf_memory = cp.transpose(img_cf_shape_cf_memory, axes=(1, 2, 0))
        img_opencv = as_opencv(
            img_cl_shape_cf_memory,
            device=target_device
        )
        np.testing.assert_almost_equal(
            img_opencv if isinstance(img_opencv, np.ndarray) else img_opencv.download(),
            img_cl_shape_cf_memory.get()
        )

    @pytest.mark.parametrize("input_type", [np.uint8, np.float32])
    @pytest.mark.parametrize("target_device", ['cuda', 'cpu', None])
    def test_cupy_last_channel_memory(self, input_type, target_device):

        # shape - [height, width, channels]
        np_image = np.random.randint(0, 255, (10, 20, 3)).astype(input_type)
        img_cl_shape_cl_memory = cupy.asarray(np_image)

        img_opencv = as_opencv(
            img_cl_shape_cl_memory,
            device=target_device
        )
        np.testing.assert_almost_equal(
            img_opencv if isinstance(img_opencv, np.ndarray) else img_opencv.download(),
            np_image
        )

        # shape - [height, width, channels]. memory_format=torch.channel_last
        img_cf_shape_cl_memory = cp.transpose(img_cl_shape_cl_memory, axes=(2, 0, 1))
        img_opencv = as_opencv(
            img_cf_shape_cl_memory,
            input_format='channel_first',
            device=target_device
        )
        np.testing.assert_almost_equal(
            img_opencv if isinstance(img_opencv, np.ndarray) else img_opencv.download(),
            np_image
        )


class TestToTorch:

    @pytest.mark.parametrize("input_device", ['cuda', 'cpu'])
    @pytest.mark.parametrize("input_type", [np.uint8, np.float32])
    @pytest.mark.parametrize("target_device", ['cuda', 'cpu', None])
    def test_opencv_memory(
        self,
        input_device,
        input_type,
        target_device
    ):
        # shape - [height, width, channels]
        if input_device == 'cpu':
            image = np.random.randint(0, 255, (10, 20, 3)).astype(input_type)
        elif input_device == 'cuda':
            image = cv2.cuda_GpuMat()
            image.upload(np.random.randint(0, 255, (10, 20, 3)).astype(input_type))
        else:
            raise ValueError(f"Unsupported input device {input_device}")

        img_torch = as_pytorch(image, device=target_device)