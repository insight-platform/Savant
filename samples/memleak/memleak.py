"""Analytics module."""
import numpy as np
import cv2

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.deepstream.opencv_utils import nvds_to_gpu_mat


class MemLeak(NvDsPyFuncPlugin):

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        noise_width = 250
        noise_height = 250
        start_top = 200
        start_left = 200

        noise = np.random.rand(noise_height, noise_width, 3) * 255
        noise = noise.astype(np.uint8)
        alpha = np.empty((noise_height, noise_width, 1), dtype=np.uint8)
        alpha.fill(128)
        noise = np.concatenate((noise, alpha), axis=2)

        stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:

            overlay = cv2.cuda.GpuMat(noise_height, noise_width, cv2.CV_8UC4)
            overlay.upload(noise, stream)

            row_range = start_top, start_top + noise_height
            col_range = start_left, start_left + noise_width
            frame_roi = cv2.cuda.GpuMat(frame_mat, row_range, col_range)

            cv2.cuda.alphaComp(overlay, frame_roi, cv2.cuda.ALPHA_OVER, frame_roi, stream=stream)
