from typing import Union, Optional

import cv2
import cupy as cp
import numpy as np


TYPE_MAP_OPENCV = {
    cv2.CV_8U: "u1", cv2.CV_8UC3: "u1", cv2.CV_8UC4: "u1",
    cv2.CV_8S: "i1", cv2.CV_8SC3: "i1", cv2.CV_8SC4: "i1",
    cv2.CV_16U: "u2", cv2.CV_16UC3: "u2", cv2.CV_16UC4: "u2",
    cv2.CV_16S: "i2", cv2.CV_16SC3: "i2", cv2.CV_16SC4: "i2",
    cv2.CV_32S: "i4", cv2.CV_32SC3: "i4", cv2.CV_32SC4: "i4",
    cv2.CV_32F: "f4", cv2.CV_32FC3: "f4", cv2.CV_32FC4: "f4",
    cv2.CV_64F: "f8", cv2.CV_64FC3: "f8", cv2.CV_64FC4: "f8",
}

TYPE_MAP_NUMPY_TO_OPENCV = {
    1: {
        "|u1": cv2.CV_8U, "|u2": cv2.CV_16U,
        "|i1": cv2.CV_8S, "|i2": cv2.CV_16S, "|i4": cv2.CV_32S,
        "|f4": cv2.CV_32F, "|f8": cv2.CV_64F,
    },
    3: {
        "|u1": cv2.CV_8UC3, "|u2": cv2.CV_16UC3,
        "|i1": cv2.CV_8SC3, "|i2": cv2.CV_16SC3, "|i4": cv2.CV_32SC3,
        "<f4": cv2.CV_32FC3, "<f8": cv2.CV_64FC3,
    },
    4: {
        "|u1": cv2.CV_8UC4, "|u2": cv2.CV_16UC4,
        "|i1": cv2.CV_8SC4, "|i2": cv2.CV_16SC4, "|i4": cv2.CV_32SC4,
        "<f4": cv2.CV_32FC4, "<f8": cv2.CV_64FC4,
    }
}


def opencv_to_numpy_type(opencv_type):
    return TYPE_MAP_OPENCV[opencv_type]


def numpy_type_to_opencv(numpy_type, channels):
    if channels in TYPE_MAP_NUMPY_TO_OPENCV:
        if numpy_type in TYPE_MAP_NUMPY_TO_OPENCV[channels]:
            return TYPE_MAP_NUMPY_TO_OPENCV[channels][numpy_type]
        else:
            raise ValueError(f"Unsupported numpy type {numpy_type} for {channels} channels."
                             f"Supported types are {TYPE_MAP_NUMPY_TO_OPENCV[channels].keys()}")
    raise ValueError(f"Unsupported number of channels {channels}. "
                     f"Supported channels are {TYPE_MAP_NUMPY_TO_OPENCV.keys()}")


class OpenCVGpuMatWrapper:
    def __init__(self, gpu_mat: cv2.cuda.GpuMat):
        w, h = gpu_mat.size()
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": (h, w, gpu_mat.channels()),
            "data": (gpu_mat.cudaPtr(), False),
            "typestr": opencv_to_numpy_type(gpu_mat.type()),
            "strides": (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()),
        }


def as_pytorch(
        img: Union[cv2.cuda.GpuMat, cp.ndarray, np.ndarray],
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        device: Optional[str] = None
):
    """Converts GPU image to PyTorch tensor.

        img - is an image represented as OpenCV or CuPy array.
        input_format - is a shape format of the input image
            (`channels_first` or `channels_last`). `channels_last` is used as default
            for cupy arrays. Parameter is ignored for OpenCV images.
        output_format - is a shape format of the output image
        (`channels_first` or `channels_last`). If output_format is None the format is
            the same as format of the input image.
        device - is a device ('cpu' or 'cuda') on which the PyTorch tensor
            will be located.
    """
    import torch
    if isinstance(img, cp.ndarray):
        if device is None:
            device = "cuda"
        torch_img = torch.as_tensor(img, device=device)
        if input_format == "channels_last" and (output_format == "channels_first" or output_format is None):
            torch_img = torch_img.permute(2, 0, 1)
        elif input_format == "channels_first" and output_format == "channels_last":
            torch_img = torch_img.permute(1, 2, 0)
        return torch_img
    elif isinstance(img, cv2.cuda.GpuMat):
        if device is None:
            device = "cuda"
        torch_img = torch.as_tensor(OpenCVGpuMatWrapper(img), device=device)
        if output_format == "channels_first" or output_format is None:
            torch_img = torch_img.permute(2, 0, 1)
        return torch_img
    elif isinstance(img, np.ndarray):
        if device is None:
            device = "cpu"
        torch_img = torch.as_tensor(img, device=device)
        if output_format == "channels_first":
            torch_img = torch_img.permute(2, 0, 1)
        return torch_img

    raise TypeError(f"Unsupported type {type(img)} to convert into PyTorch tensor.")


def as_opencv(
        img,
        device: Optional[str] = None,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
):
    """Converts GPU image to OpenCV GPU image.
    img - is a GPU image represented as PyTorch tensor or CuPy array.
        device - is a device ('cpu' or 'cuda') on which the opencv image will be located.
        If device is None, the image is converted to the same device as the input image.
    input_format - is a shape format of the input image. `channels_first` or
        `channels_last`. If the input image is a PyTorch tensor,
        the default format is 'channels_first'. If the input image
        is a CuPy array, the format is 'channels_last'.
    """
    try:
        import torch
        torch_imported = True
    except ImportError:
        torch_imported = False

    channel_index = None
    height_index = None
    width_index = None
    if input_format == "channel_first":
        channel_index = 0
        height_index = 1
        width_index = 2
    elif input_format == "channel_last":
        channel_index = 2
        height_index = 0
        width_index = 1

    if torch_imported and isinstance(img, torch.Tensor):
        if channel_index is None:
            channel_index = 0
            height_index = 1
            width_index = 2

        if img.dim() != 3:
            raise ValueError(f"Unsupported shape {img.shape} of PyTorch tensor. "
                             f"Only 3D tensors are supported to convert into image.")
        if img.shape[channel_index] > 4 or img.shape[channel_index] == 2:
            raise ValueError(
                f"Unsupported number of channels {img.shape[0]} of PyTorch tensor. "
                f"Only 1, 3, 4 channels image are supported.")
        if device is None:
            device = img.device
        else:
            device = torch.device(device)

        if device.type == 'cuda':
            if not img.is_cuda:
                img = img.cuda()

            np_type = numpy_type_to_opencv(
                numpy_type=img.__cuda_array_interface__['typestr'],
                channels=img.__cuda_array_interface__['shape'][channel_index]
            )

            cuda_int = img.__cuda_array_interface__
            if img.stride()[channel_index] != int(cuda_int["typestr"][-1]) or \
                    not img.is_contiguous(memory_format=torch.channels_last):
                if channel_index == 0:
                    img = img.permute(1, 2, 0)
                img = img.ravel()

            return cv2.cuda.createGpuMatFromCudaMemory(
                (cuda_int["shape"][width_index], cuda_int["shape"][height_index]),
                np_type,
                img.__cuda_array_interface__['data'][0],
            )

        elif device.type == "cpu":
            if channel_index == 2:
                return img.cpu().numpy()
            return img.permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Unsupported device {device} to convert into OpenCV GPU image.")

    elif isinstance(img, cp.ndarray):
        if channel_index is None:
            channel_index = 2
            height_index = 0
            width_index = 1
        if img.ndim != 3:
            raise ValueError(f"Unsupported shape {img.shape} of CuPy array. "
                             f"Only 3D arrays are supported to convert into image.")
        if img.shape[channel_index] > 4 or img.shape[channel_index] == 2:
            raise ValueError(
                f"Unsupported number of channels {img.shape[channel_index]} of CuPy array. "
                f"Only 1, 3, 4 channels image are supported.")

        if device is None:
            device = 'cuda'

        if device == "cuda":
            np_type = numpy_type_to_opencv(
                numpy_type=img.__cuda_array_interface__['typestr'],
                channels=img.__cuda_array_interface__['shape'][channel_index]
            )
            cuda_int = img.__cuda_array_interface__
            if img.strides[channel_index] != int(img.__cuda_array_interface__['typestr'][-1]):
                if channel_index == 0:
                    img = img.transpose(1, 2, 0)
                img = img.ravel()

            return cv2.cuda.createGpuMatFromCudaMemory(
                (cuda_int["shape"][width_index], cuda_int["shape"][height_index]),
                np_type,
                img.__cuda_array_interface__['data'][0],
            )

        elif device == "cpu":
            if channel_index == 2:
                return img.get()
            return cp.transpose(img, (1, 2, 0)).get()
        else:
            raise TypeError(f"Unsupported device {device} to convert into OpenCV GPU image.")

    raise TypeError(f"Unsupported type {type(img)} to convert into OpenCV GPU image.")


def as_cupy(
        img,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
) -> cp.ndarray:
    """Converts PyTorch or OpenCv image to CuPy GPU image.

    img - is a image in PyTorch or OpenCV format.
    input_format - is a shape format of the input image
        (`channels_first` or `channels_last`). `channels_last` is used as default
        for cupy arrays. Parameter is ignored for OpenCV images.
    output_format - is a shape format of the output image
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
        if (input_format == "channels_last" or input_format is None) \
                and output_format == "channels_first":
            cupy_image = cp.transpose(cupy_image, (2, 0, 1))
        elif input_format == "channels_first" and output_format == "channels_last":
            cupy_image = cp.transpose(cupy_image, (1, 2, 0))
        return cupy_image
    elif torch_imported and isinstance(img, torch.Tensor):
        cupy_image = cp.asarray(img)
        if input_format == "channels_last" and output_format == "channels_first":
            cupy_image = cp.transpose(cupy_image, (2, 0, 1))
        elif input_format == "channels_first" \
                and (output_format == "channels_last" or output_format is None):
            cupy_image = cp.transpose(cupy_image, (1, 2, 0))
        return cupy_image
    elif isinstance(img, cv2.cuda.GpuMat):
        cupy_image = cp.asarray(OpenCVGpuMatWrapper(img))
        if output_format == "channels_first":
            cupy_image = cp.transpose(cupy_image, (2, 0, 1))
        return cupy_image
    raise TypeError(f"Unsupported type {type(img)} to convert into CuPy GPU image.")