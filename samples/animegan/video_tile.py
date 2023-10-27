"""Background remover module."""
import cv2
import numpy as np
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.utils.artist import Artist


class VideoTile(NvDsPyFuncPlugin):
    """Background remover pyfunc.

    The class is designed to process video frame metadata and remove the background from the frame.
    MOG2 method from openCV is used to remove background.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        for obj in frame_meta.objects:
            if obj.is_primary:
                attr = obj.get_attr_meta("v2_paprika", "gan")
                if attr:
                    img = attr.value


        stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            # self.logger.info('host shape %s, gpu shape %s', img.shape, frame_mat.size())
            frame_mat.upload(img, stream)

            # with Artist(frame_mat, stream) as artist:

                # ref_frame = cv2.cuda_GpuMat(
                #     frame_mat,
                #     (0, 0, int(frame_meta.roi.width), int(frame_meta.roi.height)),
                # )
                # cropped = ref_frame.clone()

                # cu_mat_fg = back_sub.apply(cropped, -1, stream)
                # res_image = ref_frame.copyTo(cu_mat_fg, stream)
                # artist.add_graphic(res_image, (int(frame_meta.roi.width), 0))
