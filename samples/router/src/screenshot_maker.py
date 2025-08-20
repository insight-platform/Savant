import cupy as cp
import cv2
from nvidia import nvimgcodec
from savant_rs.logging import LogLevel, log

from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.gstreamer import Gst
from savant.utils.memory_repr import opencv_gpu_mat_as_cupy_array
from savant.utils.platform import is_aarch64


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not is_aarch64():
            self.codec = nvimgcodec.Encoder(device_id=0)
        else:
            self.codec = nvimgcodec.Encoder(
                backend_kinds=[nvimgcodec.BackendKind.HW_GPU_ONLY]
            )

        self.encode_params = nvimgcodec.EncodeParams(
            quality_type=nvimgcodec.QualityType.QUALITY, quality_value=95
        )

    def draw(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        uuid = frame_meta.video_frame.uuid
        cuda_stream = self.get_cuda_stream(frame_meta)
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            log(
                LogLevel.Info,
                'overlay',
                f'Making a screenshot for stream {frame_meta.source_id} frame {uuid}',
            )
            rgb_image = cv2.cuda.cvtColor(frame_mat, cv2.COLOR_RGBA2RGB)
            cupy_image = opencv_gpu_mat_as_cupy_array(rgb_image)  # zero copy
            self.codec.write(
                f'/screenshots/{frame_meta.source_id}-{uuid}.jpg',
                nvimgcodec.as_image(cp.ascontiguousarray(cupy_image)),
                'jpeg',
                params=self.encode_params,
                cuda_stream=cuda_stream.cudaPtr(),
            )
