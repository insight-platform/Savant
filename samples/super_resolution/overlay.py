"""Overlay."""

import cv2
import numpy as np

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.parameter_storage import param_storage
from savant.utils.artist import Artist

SR_MODEL_NAME = param_storage()['sr_model']
SR_ATTR_NAME = 'sr_frame'
INPUT_RESOLUTION = (
    param_storage()['frame']['width'],
    param_storage()['frame']['height'],
)
SUPER_RESOLUTION = (
    param_storage()['frame']['width'] * param_storage()['sr_scale'],
    param_storage()['frame']['height'] * param_storage()['sr_scale'],
)


class SROverlay(NvDsPyFuncPlugin):
    """Super resolution overlay pyfunc."""

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        cuda_stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat, Artist(
            frame_mat, cuda_stream
        ) as artist:
            # TODO: original + super resolution mix
            sr_lt = (0, 0)  # super resolution left, top

            # place origin, then super resolution
            if frame_mat.size()[0] > SUPER_RESOLUTION[0]:
                # scale original image and place first
                source_image = cv2.cuda_GpuMat(
                    frame_mat,
                    (0, 0, INPUT_RESOLUTION[0], INPUT_RESOLUTION[1]),
                )
                scaled_image = cv2.cuda.resize(
                    src=source_image,
                    dsize=SUPER_RESOLUTION,
                    # interpolation=cv2.INTER_LINEAR,
                    stream=cuda_stream,
                )
                artist.add_graphic(scaled_image, (0, 0))
                sr_lt = (scaled_image.size()[0], 0)

            # check super resolution attr
            sr_attr = None
            for obj_meta in frame_meta.objects:
                if obj_meta.is_primary:
                    sr_attr = obj_meta.get_attr_meta(SR_MODEL_NAME, SR_ATTR_NAME)
                    break

            # transform super resolution and place on the frame
            if sr_attr:
                sr_image_np = sr_attr.value.clip(0.0, 1.0)
                sr_image_np = (sr_image_np * 255).astype(np.uint8)
                # chw => hwc
                sr_image_np = np.transpose(sr_image_np, (1, 2, 0))
                # rgb => rgba
                sr_image_np = np.dstack(
                    (sr_image_np, np.full(SUPER_RESOLUTION[::-1], 255, dtype=np.uint8))
                )
                artist.add_graphic(sr_image_np, sr_lt)
