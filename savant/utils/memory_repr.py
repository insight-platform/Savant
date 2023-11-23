import cupy as cp
import cv2

__all__ = [
    'opencv_gpu_mat_as_cupy',
    'cupy_as_opencv_gpu_mat',
]

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


def _numpy_type_to_opencv_mat_type(numpy_type: str, channels: int) -> int:
    depth = NUMPY_TYPE_TO_OPENCV.get(numpy_type, None)
    if depth is None:
        raise TypeError(f'Unsupported type {numpy_type} to convert into OpenCV type.')
    # equivalent to unexposed opencv C++ macro CV_MAKETYPE(depth,channels):
    return depth + ((channels - 1) << 3)


class OpenCVGpuMatCudaArrayInterface:
    """OpenCV GpuMat __cuda_array_interface__."""

    def __init__(self, gpu_mat: cv2.cuda.GpuMat):
        width, height = gpu_mat.size()
        channels = gpu_mat.channels()
        depth = gpu_mat.depth()
        type_str = OPENCV_TYPE_TO_NUMPY.get(depth)
        assert type_str is not None, 'Unsupported OpenCV GpuMat type.'
        self.__cuda_array_interface__ = {
            'version': 3,
            'shape': (height, width, channels) if channels > 1 else (height, width),
            'data': (gpu_mat.cudaPtr(), False),
            'typestr': type_str,
            'descr': [('', type_str)],
            # 'stream': 1,  # TODO: Investigate
            'strides': (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1())
            if channels > 1
            else (gpu_mat.step, gpu_mat.elemSize()),
        }


def opencv_gpu_mat_as_cupy(gpu_mat: cv2.cuda.GpuMat) -> cp.ndarray:
    """Returns CuPy ndarray for specified OpenCV GpuMat."""
    return cp.asarray(OpenCVGpuMatCudaArrayInterface(gpu_mat))


def cupy_as_opencv_gpu_mat(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    """Returns OpenCV GpuMat for specified CuPy ndarray.
    Supports 2 and 3 dims arrays in HWC format for shape and `channel_last`
    memory format only. (OpenCV format).
    """
    if arr.ndim not in (2, 3):
        raise ValueError('CuPy array must have 2 or 3 dimensions.')
    channels = 1 if len(arr.shape) == 2 else arr.shape[2]
    return cv2.cuda.createGpuMatFromCudaMemory(
        arr.__cuda_array_interface__['shape'][1::-1],
        _numpy_type_to_opencv_mat_type(arr.dtype.str, channels),
        arr.__cuda_array_interface__['data'][0],
    )
