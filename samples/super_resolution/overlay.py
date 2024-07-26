"""Overlay."""

from typing import Dict

import cv2
import numpy as np

from savant.deepstream import opencv_utils
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.parameter_storage import param_storage

SR_MODEL_NAME = param_storage()['sr_model']
SR_ATTR_NAME = param_storage()['sr_attribute']
SUPER_RESOLUTION = (
    param_storage()['frame']['width'] * param_storage()['sr_scale'],
    param_storage()['frame']['height'] * param_storage()['sr_scale'],
)


class SROverlay(NvDsPyFuncPlugin):
    def __init__(
        self,
        codec_params: Dict,
        **kwargs,
    ):
        self.codec_params = codec_params
        self.result_aux_stream = None

        super().__init__(**kwargs)

    def on_source_add(self, source_id: str):
        """Initialize an auxiliary stream for super resolution result."""

        self.logger.info('Source %s added.', source_id)

        result_source_id = f'{source_id}-super-resolution'
        result_resolution = (SUPER_RESOLUTION[0] * 2, SUPER_RESOLUTION[1])
        self.logger.info(
            'Creating result auxiliary stream %s for source %s. Resolution: %s.',
            result_source_id,
            source_id,
            result_resolution,
        )
        self.result_aux_stream = self.auxiliary_stream(
            source_id=result_source_id,
            width=result_resolution[0],
            height=result_resolution[1],
            codec_params=self.codec_params,
        )

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """

        # Get CUDA stream for asynchronous processing
        cuda_stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            # Check super resolution attr
            sr_attr = None
            for obj_meta in frame_meta.objects:
                if obj_meta.is_primary:
                    sr_attr = obj_meta.get_attr_meta(SR_MODEL_NAME, SR_ATTR_NAME)
                    break

            # Transform super resolution image
            if sr_attr:
                # Normalize array values to be within the range [0.0, 1.0]
                sr_image_np = sr_attr.value.clip(0.0, 1.0)
                # Convert the normalized array to 8-bit unsigned integer format
                sr_image_np = (sr_image_np * 255).astype(np.uint8)
                # CHW => HWC
                sr_image_np = np.transpose(sr_image_np, (1, 2, 0))
                # RGB => RGBA
                sr_image_np = np.dstack(
                    (
                        sr_image_np,
                        np.full(SUPER_RESOLUTION[::-1], 255, dtype=np.uint8),
                    )
                )

                # Create frame for the auxiliary stream.
                # The frame will be sent automatically
                aux_frame, aux_buffer = self.result_aux_stream.create_frame(
                    pts=frame_meta.pts,
                    duration=frame_meta.duration,
                )
                with nvds_to_gpu_mat(aux_buffer, batch_id=0) as aux_mat:
                    # Scale the image to display it alongside the super resolution result.
                    scaled_image = cv2.cuda.resize(
                        src=frame_mat,
                        dsize=SUPER_RESOLUTION,
                        stream=cuda_stream,
                    )

                    # Place original frame and super resolution frame side by side
                    opencv_utils.alpha_comp(aux_mat, scaled_image, (0, 0), cuda_stream)
                    opencv_utils.alpha_comp(
                        aux_mat, sr_image_np, (sr_image_np.shape[1], 0), cuda_stream
                    )
            else:
                self.logger.warning('Super resolution attribute not found.')

    def on_stop(self) -> bool:
        self.result_aux_stream = None
        return super().on_stop()

    def on_source_eos(self, source_id: str):
        self.logger.info('Got EOS from source %s.', source_id)
        self.result_aux_stream.eos()
