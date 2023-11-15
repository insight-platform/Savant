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
    def __init__(self, img: cv2.cuda.GpuMat):
        w, h = img.size()
        self.__cuda_array_interface__ = {
            "version": 3,
            "shape": (img.channels(), h, w),
            "data": (img.cudaPtr(), False),
            "typestr": opencv_to_numpy_type(img.type()),
            "strides": (1, img.step, img.elemSize()),
        }
def as_pytorch(
        img: Union[cv2.cuda.GpuMat, cp.ndarray, np.ndarray],
        device: Optional[str] = None
):
    """Converts GPU image to PyTorch tensor."""
    import torch
    if isinstance(img, cp.ndarray):
        if device is None:
            device = "cuda"
        return torch.as_tensor(img, device=device)
    elif isinstance(img, cv2.cuda.GpuMat):
        if device is None:
            device = "cuda"
        return torch.as_tensor(OpenCVGpuMatWrapper(img), device=device)
    elif isinstance(img, np.ndarray):
        if device is None:
            device = "cpu"
        return torch.as_tensor(img, device=device)

    raise TypeError(f"Unsupported type {type(img)} to convert into PyTorch tensor.")


def as_opencv(img, device: Optional[str] = None ):
    """Converts GPU image to OpenCV GPU image.
    img - is a GPU image in PyTorch or CuPy format. The shape of the image is (C, H, W).
    device - is a device ('cpu' or 'cuda') to convert the image.
    If device is None, the image is converted to the same device as the input image.
    """
    import cupy as cp
    try:
        import torch
        torch_imported = True
    except ImportError:
        torch_imported = False

    if torch_imported and isinstance(img, torch.Tensor):
        import torch
        if img.dim() != 3:
            raise ValueError(f"Unsupported shape {img.shape} of PyTorch tensor. "
                             f"Only 3D tensors are supported to convert into image.")
        if img.shape[0] > 4 or img.shape[0] == 2:
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
                channels=img.__cuda_array_interface__['shape'][0]
            )

            stride = img.stride()
            if stride[0] != 1 or not img.is_contiguous(memory_format=torch.channels_last):
                img = img.unsqueeze(0).to(memory_format=torch.channels_last).squeeze(0)

            cuda_int = img.__cuda_array_interface__
            pitch = cuda_int['shape'][0] * cuda_int['shape'][2]
            return cv2.savant.createGpuMat(
                cuda_int['shape'][1],
                cuda_int['shape'][2],
                np_type,
                cuda_int['data'][0],
                pitch,
            )
        elif device.type == "cpu":
            return img.cpu().numpy()
        else:
            raise TypeError(f"Unsupported device {device} to convert into OpenCV GPU image.")

    elif isinstance(img, cp.ndarray):
        if img.ndim != 3:
            raise ValueError(f"Unsupported shape {img.shape} of CuPy array. "
                             f"Only 3D arrays are supported to convert into image.")
        if img.shape[0] > 4 or img.shape[0] == 2:
            raise ValueError(
                f"Unsupported number of channels {img.shape[0]} of CuPy array. "
                f"Only 1, 3, 4 channels image are supported.")

        if device is None:
            device = 'cuda'
        if device == "cuda":
            np_type = numpy_type_to_opencv(
                numpy_type=img.__cuda_array_interface__['typestr'],
                channels=img.__cuda_array_interface__['shape'][0]
            )
            if img.strides[0] == int(img.__cuda_array_interface__['typestr'][-1]):
                return cv2.savant.createGpuMat(
                    img.__cuda_array_interface__['shape'][1],
                    img.__cuda_array_interface__['shape'][2],
                    np_type,
                    img.__cuda_array_interface__['data'][0],
                    img.strides[1]
                )
            else:
                tmp_tensor = torch.as_tensor(img, device=device)
                tmp_tensor = tmp_tensor.unsqueeze(0).to(memory_format=torch.channels_last).squeeze(0)
                cuda_int = tmp_tensor.__cuda_array_interface__
                pitch = cuda_int['shape'][0] * cuda_int['shape'][2]
                return cv2.savant.createGpuMat(
                    cuda_int['shape'][1],
                    cuda_int['shape'][2],
                    np_type,
                    cuda_int['data'][0],
                    pitch,
                )
                # raise ValueError(f"Unsupported stride {img.strides[0]} of CuPy array in the first dimension."
                #                  f"For the type `{img.dtype}` the stride should be equal `{img.__cuda_array_interface__['typestr'][-1]}`."
                #                  f" Change the memory format to the channel last.")
        elif device == "cpu":
            return cp.asnumpy(img)
        else:
            raise TypeError(f"Unsupported device {device} to convert into OpenCV GPU image.")

    raise TypeError(f"Unsupported type {type(img)} to convert into OpenCV GPU image.")


def as_cupy(img) -> cp.ndarray:
    """Converts PyTorch or OpenCv image to CuPy GPU image.
    img - is a GPU image in PyTorch or OpenCV format. The shape of the image is (C, H, W).
    """
    try:
        import torch
        torch_imported = True
    except ImportError:
        torch_imported = False
    if isinstance(img, np.ndarray) or (torch_imported and isinstance(img, torch.Tensor)):
        return cp.asarray(img)
    elif isinstance(img, cv2.cuda.GpuMat):
        return cp.asarray(OpenCVGpuMatWrapper(img))
    raise TypeError(f"Unsupported type {type(img)} to convert into CuPy GPU image.")