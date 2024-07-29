"""Background remover module."""

from typing import Dict

import cv2

from savant.deepstream.auxiliary_stream import AuxiliaryStream
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.parameter_storage import param_storage
from savant.utils.artist import Artist


class BgRemover(NvDsPyFuncPlugin):
    """Background remover pyfunc.

    The class is designed to process video frame metadata and remove the background from the frame.
    MOG2 method from openCV is used to remove background.
    """

    def __init__(
        self,
        codec_params: Dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codec_params = codec_params
        self.result_aux_streams: Dict[str, AuxiliaryStream] = {}
        self.back_subtractors = {}

        self.gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC4, cv2.CV_8UC4, (9, 9), 2
        )

    def on_source_add(self, source_id: str):
        """Initialize an auxiliary stream for background removal result."""

        self.logger.info('Source %s added.', source_id)
        if source_id in self.result_aux_streams:
            self.logger.info('Source %s already has a result stream.', source_id)
            return

        result_source_id = f'{source_id}-processed'
        result_resolution = (
            param_storage()['frame']['width'] * 2,
            param_storage()['frame']['height'],
        )
        self.logger.info(
            'Creating result auxiliary stream %s for source %s. Resolution: %s.',
            result_source_id,
            source_id,
            result_resolution,
        )
        self.result_aux_streams[source_id] = self.auxiliary_stream(
            source_id=result_source_id,
            width=result_resolution[0],
            height=result_resolution[1],
            codec_params=self.codec_params,
        )

        if source_id in self.back_subtractors:
            self.logger.info(
                'Source %s already has a background subtractor.', source_id
            )
            return

        self.logger.info('Creating background subtractor for source %s.', source_id)
        self.back_subtractors[source_id] = cv2.cuda.createBackgroundSubtractorMOG2()

    def on_stop(self) -> bool:
        self.result_aux_streams = {}
        return super().on_stop()

    def on_source_eos(self, source_id: str):
        """On source EOS event callback."""
        if source_id is self.back_subtractors:
            self.back_subtractors.pop(source_id)
        if source_id in self.result_aux_streams:
            self.result_aux_streams.get(source_id).eos()

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        """Process frame metadata.

        :param buffer: Gstreamer buffer with this frame's data.
        :param frame_meta: This frame's metadata.
        """
        stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            result_stream = self.result_aux_streams[frame_meta.source_id]
            # Created frame will be sent automatically
            result_frame, result_buffer = result_stream.create_frame(
                pts=frame_meta.pts,
                duration=frame_meta.duration,
            )
            with nvds_to_gpu_mat(result_buffer, batch_id=0) as result_mat:
                with Artist(result_mat, stream) as artist:
                    frame_mat_copy = frame_mat.clone()

                    back_sub = self.back_subtractors[frame_meta.source_id]
                    self.gaussian_filter.apply(frame_mat_copy, frame_mat_copy, stream=stream)
                    cu_mat_fg = back_sub.apply(frame_mat_copy, -1, stream)
                    res_image = frame_mat_copy.copyTo(cu_mat_fg, stream)

                    artist.add_graphic(frame_mat, (0, 0))
                    artist.add_graphic(res_image, (int(frame_meta.roi.width), 0))
