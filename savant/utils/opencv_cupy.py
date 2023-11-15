"""Zero-copy data exchange between CuPy and OpenCV GpuMat.
https://github.com/rapidsai/cucim/issues/329

TODO: Add tests.
"""
import cupy as cp
import cv2

__all__ = ['cupy_to_opencv', 'opencv_to_cupy']


OPENCV_CUPY_TYPE_MAP = {
    cv2.CV_8U: '|u1',
    cv2.CV_8S: '|i1',
    cv2.CV_16U: '<u2',
    cv2.CV_16S: '<i2',
    cv2.CV_32S: '<i4',
    cv2.CV_32F: '<f4',
    cv2.CV_64F: '<f8',
}
CUPY_OPENCV_TYPE_MAP = {v: k for k, v in OPENCV_CUPY_TYPE_MAP.items()}


def cupy_to_opencv(arr: cp.ndarray) -> cv2.cuda.GpuMat:
    """Returns OpenCv GpuMat for specified CuPy ndarray."""
    assert len(arr.shape) in (2, 3), 'CuPy array must have 2 or 3 dimensions.'

    depth = CUPY_OPENCV_TYPE_MAP.get(arr.dtype.str)
    assert depth is not None, 'Unsupported CuPy array type.'

    channels = 1 if len(arr.shape) == 2 else arr.shape[2]

    # equivalent to unexposed opencv C++ macro CV_MAKETYPE(depth,channels):
    mat_type = depth + ((channels - 1) << 3)

    return cv2.cuda.createGpuMatFromCudaMemory(
        arr.__cuda_array_interface__['shape'][1::-1],
        mat_type,
        arr.__cuda_array_interface__['data'][0],
    )


class OpenCVGpuMatCudaArrayInterface:
    """OpenCv GpuMat __cuda_array_interface__."""

    def __init__(self, gpu_mat: cv2.cuda.GpuMat):
        width, height = gpu_mat.size()
        channels = gpu_mat.channels()
        depth = gpu_mat.depth()

        type_str = OPENCV_CUPY_TYPE_MAP.get(depth)
        assert type_str is not None, 'Unsupported OpenCV GpuMat type.'

        self.__cuda_array_interface__ = {
            'version': 3,
            'shape': (height, width, channels) if channels > 1 else (height, width),
            'typestr': type_str,
            'descr': [('', type_str)],
            # 'stream': 1,  # TODO: Investigate
            'strides': (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1())
            if channels > 1
            else (gpu_mat.step, gpu_mat.elemSize()),
            'data': (gpu_mat.cudaPtr(), False),
        }


def opencv_to_cupy(gpu_mat: cv2.cuda.GpuMat) -> cp.ndarray:
    """Returns CuPy ndarray for specified OpenCv GpuMat."""

    return cp.asarray(OpenCVGpuMatCudaArrayInterface(gpu_mat))
