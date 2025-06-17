import cv2
from savant.deepstream.drawfunc import NvDsDrawFunc
from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
from savant.gstreamer import Gst
from savant_rs.logging import LogLevel, log


class Overlay(NvDsDrawFunc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def draw(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        uuid = frame_meta.video_frame.uuid
        with nvds_to_gpu_mat(buffer, frame_meta.frame_meta) as frame_mat:
            log(
                LogLevel.Info,
                "overlay",
                f"Making a screenshot for stream {frame_meta.source_id} frame {uuid}",
            )
            cpu_mat = frame_mat.download()
            bgr_image = cv2.cvtColor(cpu_mat, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(
                f"/screenshots/{frame_meta.source_id}-{uuid}.jpg",
                bgr_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            )