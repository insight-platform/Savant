from typing import Optional, Union

import cupy as cp
import cv2
import numpy as np

OPENCV_TYPE_TO_NUMPY = {
    cv2.CV_8U: '|u1',
    cv2.CV_8S: '|i1',
    cv2.CV_16U: '<u2',
    cv2.CV_16S: '<i2',
    cv2.CV_32S: '<i4',
    cv2.CV_32F: '<f4',
    cv2.CV_64F: '<f8',
}

NUMPY_TYPE_TO_OPENCV = {v: k for k, v in OPENCV_TYPE_TO_NUMPY.items()}


def numpy_type_to_opencv(numpy_type, channels):
    depth = NUMPY_TYPE_TO_OPENCV.get(numpy_type, None)
    if depth is None:
        raise TypeError(f'Unsupported type {numpy_type} to convert into OpenCV type.')
    return depth + ((channels - 1) << 3)


class OpenCVGpuMatWrapper:
    def __init__(self, gpu_mat: cv2.cuda.GpuMat):
        width, height = gpu_mat.size()
        channels = gpu_mat.channels()
        depth = gpu_mat.depth()
        type_str = OPENCV_TYPE_TO_NUMPY.get(depth)
        self.__cuda_array_interface__ = {
            'version': 3,
            'shape': (height, width, channels) if channels > 1 else (height, width),
            'data': (gpu_mat.cudaPtr(), False),
            'typestr': type_str,
            'strides': (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1())
            if channels > 1
            else (gpu_mat.step, gpu_mat.elemSize()),
        }


def as_pytorch(
    img: Union[cv2.cuda.GpuMat, cp.ndarray, np.ndarray],
    input_format: Optional[str] = None,
    output_format: Optional[str] = None,
    device: Optional[str] = None,
):
    """Converts GPU image to PyTorch tensor.

    :param img: is an image represented as OpenCV or CuPy array.
    :param input_format: is a shape format of the input image
        (`channels_first` or `channels_last`). `channels_last` is used as default
        for cupy arrays. Parameter is ignored for OpenCV images and grayscale images.
    :param output_format: is a shape format of the output image
        (`channels_first` or `channels_last`). If output_format is None the format is
            the same as format of the input image.
    :param device: is a device ('cpu' or 'cuda') on which the PyTorch tensor
            will be located.
    """
    import torch

    if isinstance(img, cp.ndarray):
        if device is None:
            device = 'cuda'
        torch_img = torch.as_tensor(img, device=device)

        if img.ndim == 2:
            return torch_img

        if input_format == 'channels_last' and (
            output_format == 'channels_first' or output_format is None
        ):
            torch_img = torch_img.permute(2, 0, 1)
        elif input_format == 'channels_first' and output_format == 'channels_last':
            torch_img = torch_img.permute(1, 2, 0)
        return torch_img

    elif isinstance(img, cv2.cuda.GpuMat):
        if device is None:
            device = 'cuda'

        torch_img = torch.as_tensor(OpenCVGpuMatWrapper(img), device=device)

        if img.channels() == 1:
            return torch_img

        if output_format == 'channels_first' or output_format is None:
            torch_img = torch_img.permute(2, 0, 1)
        return torch_img
    elif isinstance(img, np.ndarray):
        if device is None:
            device = 'cpu'

        torch_img = torch.as_tensor(img, device=device)

        if img.ndim == 2:
            return torch_img

        if output_format == 'channels_first':
            torch_img = torch_img.permute(2, 0, 1)
        return torch_img

    raise TypeError(f'Unsupported type {type(img)} to convert into PyTorch tensor.')


def as_opencv(
    img, input_format: Optional[str] = None, device: Optional[str] = None
) -> Union[cv2.cuda.GpuMat, np.ndarray]:
    """Converts GPU image to OpenCV GPU image.

    :param img: is a GPU image represented as PyTorch tensor or CuPy array.
    :param input_format: is a shape format of the input image. `channels_first` or
        `channels_last`. If the input image is a PyTorch tensor,
        the default format is 'channels_first'. If the input image
        is a CuPy array, the format is 'channels_last'.
    :param device: is a device ('cpu' or 'cuda') on which the opencv
        image will be located. If device is None, the image is converted
        to the same device as the input image.
    """
    try:
        import torch

        torch_imported = True
    except ImportError:
        torch_imported = False

    channel_index = None
    height_index = None
    width_index = None
    if input_format == 'channel_first':
        channel_index = 0
        height_index = 1
        width_index = 2
    elif input_format == 'channel_last':
        channel_index = 2
        height_index = 0
        width_index = 1

    if torch_imported and isinstance(img, torch.Tensor):
        if channel_index is None:
            channel_index = 0
            height_index = 1
            width_index = 2

        if img.dim() != 3 and img.dim() != 2:
            raise ValueError(
                f'Unsupported shape {img.shape} of PyTorch tensor. '
                f'Only 3D tensors are supported to convert into image.'
            )
        if img.dim() == 3 and (
            img.shape[channel_index] > 4 or img.shape[channel_index] == 2
        ):
            raise ValueError(
                f'Unsupported number of channels {img.shape[0]} of PyTorch tensor. '
                f'Only 1, 3, 4 channels image are supported.'
            )
        if device is None:
            device = img.device
        else:
            device = torch.device(device)

        if device.type == 'cuda':
            if not img.is_cuda:
                img = img.cuda()

            cuda_interface = img.__cuda_array_interface__

            if img.dim() == 2:
                np_type = NUMPY_TYPE_TO_OPENCV.get(
                    img.__cuda_array_interface__['typestr']
                )
                return cv2.cuda.createGpuMatFromCudaMemory(
                    (cuda_interface['shape'][1], cuda_interface['shape'][0]),
                    np_type,
                    img.__cuda_array_interface__['data'][0],
                )

            np_type = numpy_type_to_opencv(
                numpy_type=img.__cuda_array_interface__['typestr'],
                channels=img.__cuda_array_interface__['shape'][channel_index],
            )

            if img.stride()[channel_index] != int(
                cuda_interface['typestr'][-1]
            ) or not img.is_contiguous(memory_format=torch.channels_last):
                if channel_index == 0:
                    img = img.permute(1, 2, 0)
                img = img.ravel()

            return cv2.cuda.createGpuMatFromCudaMemory(
                (
                    cuda_interface['shape'][width_index],
                    cuda_interface['shape'][height_index],
                ),
                np_type,
                img.__cuda_array_interface__['data'][0],
            )

        elif device.type == 'cpu':
            if img.dim() == 2:
                return img.cpu().numpy()
            if channel_index == 2:
                return img.cpu().numpy()
            return img.permute(1, 2, 0).cpu().numpy()
        raise TypeError(
            f'Unsupported device {device} to convert into OpenCV GPU image.'
        )

    elif isinstance(img, cp.ndarray):
        if channel_index is None:
            channel_index = 2
            height_index = 0
            width_index = 1
        if img.ndim != 3 and img.ndim != 2:
            raise ValueError(
                f'Unsupported shape {img.shape} of CuPy array. '
                f'Only 3D arrays are supported to convert into image.'
            )
        if img.ndim == 3 and (
            img.shape[channel_index] > 4 or img.shape[channel_index] == 2
        ):
            raise ValueError(
                f'Unsupported number of channels {img.shape[channel_index]} of CuPy array. '
                f'Only 1, 3, 4 channels image are supported.'
            )

        if device is None:
            device = 'cuda'

        if device == 'cuda':

            cuda_interface = img.__cuda_array_interface__
            if img.ndim == 2:
                np_type = NUMPY_TYPE_TO_OPENCV.get(
                    img.__cuda_array_interface__['typestr']
                )
                return cv2.cuda.createGpuMatFromCudaMemory(
                    (cuda_interface['shape'][1], cuda_interface['shape'][0]),
                    np_type,
                    img.__cuda_array_interface__['data'][0],
                )

            np_type = numpy_type_to_opencv(
                numpy_type=img.__cuda_array_interface__['typestr'],
                channels=img.__cuda_array_interface__['shape'][channel_index],
            )
            if img.strides[channel_index] != int(
                img.__cuda_array_interface__['typestr'][-1]
            ):
                if channel_index == 0:
                    img = img.transpose(1, 2, 0)
                img = img.ravel()

            return cv2.cuda.createGpuMatFromCudaMemory(
                (
                    cuda_interface['shape'][width_index],
                    cuda_interface['shape'][height_index],
                ),
                np_type,
                img.__cuda_array_interface__['data'][0],
            )

        elif device == 'cpu':
            if channel_index == 2:
                return img.get()
            return cp.transpose(img, (1, 2, 0)).get()
        else:
            raise TypeError(
                f'Unsupported device {device} to convert into OpenCV GPU image.'
            )

    raise TypeError(f'Unsupported type {type(img)} to convert into OpenCV GPU image.')


def as_cupy(
    img,
    input_format: Optional[str] = None,
    output_format: Optional[str] = None,
) -> cp.ndarray:
    """Converts PyTorch or OpenCv image to CuPy GPU image.

    :param img: is a image in PyTorch or OpenCV format.
    :param input_format: is a shape format of the input image
        (`channels_first` or `channels_last`). `channels_last` is used as default
        for cupy arrays. Parameter is ignored for OpenCV images.
    :param output_format: is a shape format of the output image
        (`channels_first` or `channels_last`). If output_format is None the format is
        the same as format of the input image.
    """
    try:
        import torch

        torch_imported = True
    except ImportError:
        torch_imported = False
    if isinstance(img, np.ndarray):
        cupy_image = cp.asarray(img)

        if img.ndim == 2:
            return cupy_image

        if (
            input_format == 'channels_last' or input_format is None
        ) and output_format == 'channels_first':
            cupy_image = cp.transpose(cupy_image, (2, 0, 1))
        elif input_format == 'channels_first' and output_format == 'channels_last':
            cupy_image = cp.transpose(cupy_image, (1, 2, 0))
        return cupy_image

    elif torch_imported and isinstance(img, torch.Tensor):
        cupy_image = cp.asarray(img)

        if img.ndim == 2:
            return cupy_image

        if input_format == 'channels_last' and output_format == 'channels_first':
            cupy_image = cp.transpose(cupy_image, (2, 0, 1))
        elif input_format == 'channels_first' and (
            output_format == 'channels_last' or output_format is None
        ):
            cupy_image = cp.transpose(cupy_image, (1, 2, 0))
        return cupy_image
    elif isinstance(img, cv2.cuda.GpuMat):

        cupy_image = cp.asarray(OpenCVGpuMatWrapper(img))

        if img.channels == 1:
            return cupy_image

        if output_format == 'channels_first':
            cupy_image = cp.transpose(cupy_image, (2, 0, 1))
        return cupy_image
    raise TypeError(f'Unsupported type {type(img)} to convert into CuPy GPU image.')
