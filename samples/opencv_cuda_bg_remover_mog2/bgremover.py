"""Background remover module."""
from savant.gstreamer import Gst
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.utils.artist import Artist
from savant.deepstream.opencv_utils import (
    nvds_to_gpu_mat,
)
import cv2

class BgRemover(NvDsPyFuncPlugin):
    """Background remover pyfunc.

    The class is designed to process video frame metadata and remove the background from the frame.
    MOG2 method from openCV is used to remove background.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stream = cv2.cuda.Stream_Null()
        self.back_subtractors = {}

        self.gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC4, cv2.CV_8UC4, (9, 9), 2
        )


    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        if source_id is self.back_subtractors:
            self.back_subtractors.pop(source_id)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        pass
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            with Artist(frame_mat) as artist:
                if frame_meta.source_id in self.back_subtractors:
                    back_sub = self.back_subtractors[frame_meta.source_id]
                else:
                    back_sub = cv2.cuda.createBackgroundSubtractorMOG2()
                    self.back_subtractors[frame_meta.source_id] = back_sub
                ref_frame = cv2.cuda_GpuMat(frame_mat, (0, 0, int(frame_meta.roi.width), int(frame_meta.roi.height)))
                cropped = ref_frame.clone()
                self.gaussian_filter.apply(cropped, cropped, stream=self.stream)
                cu_mat_fg = back_sub.apply(cropped, -1, self.stream)
                res_image = ref_frame.copyTo(cu_mat_fg, self.stream)
                artist.add_graphic(res_image, (int(frame_meta.roi.width), 0))
