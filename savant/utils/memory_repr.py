import cupy as cp
import cv2

OPENCV_TO_NUMPY_TYPE_MAP = {
    cv2.CV_8U: '|u1',
    cv2.CV_8S: '|i1',
    cv2.CV_16U: '<u2',
    cv2.CV_16S: '<i2',
    cv2.CV_32S: '<i4',
    cv2.CV_32F: '<f4',
    cv2.CV_64F: '<f8',
}

NUMPY_TO_OPENCV_TYPE_MAP = {v: k for k, v in OPENCV_TO_NUMPY_TYPE_MAP.items()}


class OpenCVGpuMatCudaArrayInterface:
    """OpenCV GpuMat __cuda_array_interface__.
    https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    """

    def __init__(self, gpu_mat: cv2.cuda.GpuMat):
        width, height = gpu_mat.size()
        channels = gpu_mat.channels()
        type_str = OPENCV_TO_NUMPY_TYPE_MAP.get(gpu_mat.depth())
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


def cuda_array_as_opencv_gpu_mat(arr) -> cv2.cuda.GpuMat:
    """Returns OpenCV GpuMat for the given cuda array.
    The array must support __cuda_array_interface__ (CuPy, PyTorch),
    have 2 or 3 dims and be in C-contiguous layout.
    """
    shape = arr.__cuda_array_interface__['shape']
    assert len(shape) in (2, 3), 'Array must have 2 or 3 dimensions.'

    dtype = arr.__cuda_array_interface__['typestr']
    depth = NUMPY_TO_OPENCV_TYPE_MAP.get(dtype)
    assert (
        depth is not None
    ), f'Array must be of one of the following types {list(NUMPY_TO_OPENCV_TYPE_MAP)}.'

    strides = arr.__cuda_array_interface__['strides']
    assert strides is None, 'Array must be in C-contiguous layout.'

    channels = 1 if len(shape) == 2 else shape[2]
    # equivalent to unexposed opencv C++ macro CV_MAKETYPE(depth,channels):
    mat_type = depth + ((channels - 1) << 3)

    return cv2.cuda.createGpuMatFromCudaMemory(
        shape[1::-1], mat_type, arr.__cuda_array_interface__['data'][0]
    )


def opencv_gpu_mat_as_cupy_array(gpu_mat: cv2.cuda.GpuMat) -> cp.ndarray:
    """Returns CuPy ndarray in HWC format for the given OpenCV GpuMat (zero-copy)."""
    return cp.asarray(OpenCVGpuMatCudaArrayInterface(gpu_mat))


def cupy_array_as_opencv_gpu_mat(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    """Returns OpenCV GpuMat for the given CuPy ndarray (zero-copy).
    The array must have 2 or 3 dims in HWC format and C-contiguous layout.

    Use `cupy.shape` and `cupy.strides` to check if an array
    has supported shape format and is contiguous in memory.

    Use `cupy.transpose()` and `cupy.ascontiguousarray()` to transform an array
    if necessary (creates a copy of the array).
    """
    return cuda_array_as_opencv_gpu_mat(arr)
