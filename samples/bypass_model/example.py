"""Example of a Python plugin that compares multiple module outputs."""

from typing import Dict

import cupy as cp

from savant.deepstream import opencv_utils
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.parameter_storage import param_storage
from savant.utils.memory_repr import cupy_array_as_opencv_gpu_mat

ELEMENT_NAME = param_storage()['element_name']
ATTR_NAME = param_storage()['attribute_name']
RESULT_RESOLUTION = (1080, 720)  # todo move to params
result_resolution = (RESULT_RESOLUTION[0] * 2, RESULT_RESOLUTION[1])


class CompareOutputs(NvDsPyFuncPlugin):  # todo update class name and description
    """Compare raw and pre-processed images and print the result to log."""

    def __init__(
        self,
        codec_params: Dict,
        **kwargs,
    ):
        self.codec_params = codec_params
        self.result_aux_stream = None

        super().__init__(**kwargs)

    def on_source_add(self, source_id: str):
        """Initialize an auxiliary stream for result."""

        self.logger.info('Source %s added.', source_id)

        result_source_id = f'{source_id}-with-preprocessed-frame'
        self.logger.info(
            'Creating result stream %s for source %s. Resolution: %s.',
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
            element_attr = None
            for obj_meta in frame_meta.objects:
                if obj_meta.is_primary:
                    element_attr = obj_meta.get_attr_meta(ELEMENT_NAME, ATTR_NAME)
                    break

            # Transform super resolution image
            if element_attr:
                # CHW => HWC
                preprocessed_image = cp.transpose(element_attr.value, (1, 2, 0))
                # RGB => RGBA
                preprocessed_image = cp.dstack(
                    (
                        preprocessed_image.astype(cp.uint8),
                        cp.full(RESULT_RESOLUTION[::-1], 255, dtype=cp.uint8),
                    )
                )
                start_point = (preprocessed_image.shape[1], 0)
                preprocessed_image_mat = cupy_array_as_opencv_gpu_mat(
                    preprocessed_image
                )

                # Create frame for the auxiliary stream.
                # The frame will be sent automatically
                aux_frame, aux_buffer = self.result_aux_stream.create_frame(
                    pts=frame_meta.pts,
                    duration=frame_meta.duration,
                )
                with nvds_to_gpu_mat(aux_buffer, batch_id=0) as aux_mat:
                    # Create a background image
                    white_image = cp.full((result_resolution[1], result_resolution[0], 4), 255, dtype=cp.uint8)
                    background = cupy_array_as_opencv_gpu_mat(white_image)
                    opencv_utils.alpha_comp(aux_mat, background, (0, 0), stream=cuda_stream)

                    # Place original frame and pre-processed frame side by side
                    opencv_utils.alpha_comp(
                        aux_mat,
                        frame_mat,
                        (0, 0),
                        stream=cuda_stream,
                    )
                    opencv_utils.alpha_comp(
                        aux_mat,
                        preprocessed_image_mat,
                        start_point,
                        stream=cuda_stream,
                    )
            else:
                self.logger.warning(
                    'Model %s attribute %s not found.', ELEMENT_NAME, ATTR_NAME
                )

    def on_stop(self) -> bool:
        self.result_aux_stream = None
        return super().on_stop()

    def on_source_eos(self, source_id: str):
        self.logger.info('Got EOS from source %s.', source_id)
        self.result_aux_stream.eos()
