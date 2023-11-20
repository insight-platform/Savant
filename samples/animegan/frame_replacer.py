"""FrameReplacer module."""

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.parameter_storage import param_storage

GAN_NAME = param_storage()['gan_name']


class FrameReplacer(NvDsPyFuncPlugin):
    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        for obj in frame_meta.objects:
            if obj.is_primary:
                attr = obj.get_attr_meta(GAN_NAME, 'gan')
                break

        if attr:
            img = attr.value
            stream = self.get_cuda_stream(frame_meta)
            with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
                frame_mat.upload(img, stream)
