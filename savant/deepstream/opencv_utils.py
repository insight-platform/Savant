"""Utils for working OpenCV representation of CUDA memory (cv2.cuda.GpuMat)."""

from contextlib import contextmanager
from typing import ContextManager, Tuple, Union, Optional

import cv2
import numpy as np
import pyds
from pysavantboost import PyDSCudaMemory

from savant.gstreamer import Gst


@contextmanager
def nvds_to_gpu_mat(
    buffer: Gst.Buffer,
    nvds_frame_meta: pyds.NvDsFrameMeta,
) -> ContextManager[cv2.cuda.GpuMat]:
    """Build GpuMat header for allocated CUDA-memory of the frame.

    Usage example:

    .. code-block:: python

        with nvds_to_gpu_mat(buffer, nvds_frame_meta) as frame_mat:
            roi = cv2.cuda.GpuMat(frame_mat, (300, 400, 100, 200))
            # Fill area on the frame with red color
            roi.setTo((255, 0, 0, 255))

    :param buffer: Gstreamer buffer which contains NvBufSurface.
    :param nvds_frame_meta: NvDs frame metadata which contains frame info.
    :return: GpuMat header for allocated CUDA-memory of the frame.
    """

    py_ds_cuda_memory = PyDSCudaMemory(hash(buffer), nvds_frame_meta.batch_id)
    try:
        cuda_ptr = py_ds_cuda_memory.GetMapCudaPtr()
        yield cv2.savant.createGpuMat(
            py_ds_cuda_memory.height,
            py_ds_cuda_memory.width,
            cv2.CV_8UC4,
            cuda_ptr,
        )
    finally:
        py_ds_cuda_memory.UnMapCudaPtr()


def alpha_comp(
    frame: cv2.cuda.GpuMat,
    overlay: Union[cv2.cuda.GpuMat, np.ndarray],
    start: Tuple[int, int],
    alpha_op=cv2.cuda.ALPHA_OVER,
    stream: Optional[cv2.cuda.Stream] = None,
):
    """In place composition of two images using alpha opacity values contained in
    each image.

    Usage example:

    .. code-block:: python

        overlay = cv2.imread("overlay.png", cv2.IMREAD_UNCHANGED)
        with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
            # Place overlay on the frame
            alpha_comp(frame_mat, overlay, (300, 400))

    :param frame: Base image. Note: this image will be changed.
    :param overlay: The image to overlay.
    :param start: Top-left corner of the overlay on the frame.
    :param alpha_op: Flag specifying the alpha-blending operation.
    :param stream: CUDA stream.
    """

    if isinstance(overlay, np.ndarray):
        overlay = cv2.cuda.GpuMat(overlay)
    roi_mat = cv2.cuda.GpuMat(frame, start + overlay.size())
    cv2.cuda.alphaComp(overlay, roi_mat, alpha_op, roi_mat, stream=stream)


def apply_cuda_filter(
    cuda_filter: cv2.cuda.Filter,
    frame: cv2.cuda.GpuMat,
    roi: Tuple[int, int, int, int],
    stream: Optional[cv2.cuda.Stream] = None,
):
    """Apply CUDA filter to a frame in place.

    Usage example:

    .. code-block:: python

        gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC4, cv2.CV_8UC4, (31, 31), 100, 100
        )
        with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
            # Blur specified area of the frame with gaussian blur
            apply_cuda_filter(gaussian_filter, frame_mat, (300, 400, 100, 200))

    :param cuda_filter: CUDA filter.
    :param frame: Frame to which apply CUDA filter.
    :param roi: Region of interest on the frame: (left, top, width, height).
    :param stream: CUDA stream.
    """

    roi_mat = cv2.cuda.GpuMat(frame, roi)
    cuda_filter.apply(roi_mat, roi_mat, stream=stream)


def draw_rect(
    frame: cv2.cuda.GpuMat,
    rect: Tuple[int, int, int, int],
    color: Tuple[int, int, int, int],
    thickness: int = 1,
    stream: Optional[cv2.cuda.Stream] = None,
):
    """Draw rectangle on a frame.

    Usage example:

    .. code-block:: python

        with nvds_to_gpu_mat(gst_buffer, nvds_frame_meta) as frame_mat:
            draw_rect(frame_mat, (400, 100, 500, 300), (255, 0, 0, 255), 4)

    :param frame: The frame.
    :param rect: Rectangle coordinates (left, top, right, bottom).
    :param color: Border color (R, G, B, A).
    :param thickness: Border thickness.
    :param stream: CUDA stream.
    """

    x1, y1, x2, y2 = rect
    frame.colRange(x1, x2).rowRange(y1, y1 + thickness).setTo(color, stream=stream)
    frame.colRange(x1, x2).rowRange(y2 - thickness, y2).setTo(color, stream=stream)
    frame.colRange(x1, x1 + thickness).rowRange(y1, y2).setTo(color, stream=stream)
    frame.colRange(x2 - thickness, x2).rowRange(y1, y2).setTo(color, stream=stream)
