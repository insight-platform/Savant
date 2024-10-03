from typing import Optional, Tuple

import cv2
from savant_rs.primitives import VideoFrame

from savant.deepstream.ingress_converter import BaseIngressConverter


class IngressConverter(BaseIngressConverter):
    def convert(
        self,
        source_id: str,
        meta: VideoFrame,
        in_frame: cv2.cuda.GpuMat,
        out_frame: cv2.cuda.GpuMat,
    ):
        self.logger.info(
            "Converting frame from %s: %sx%s -> %sx%s",
            source_id,
            *in_frame.size(),
            *out_frame.size()
        )
        cv2.cuda.resize(src=in_frame, dst=out_frame, dsize=out_frame.size())

    def on_stream_start(
        self, source_id: str, width: int, height: int
    ) -> Optional[Tuple[int, int]]:
        self.logger.info(
            "Stream %s started with resolution %sx%s", source_id, width, height
        )
        return 1280, 720

    def on_stream_stop(self, source_id: str) -> None:
        self.logger.info("Stream %s stopped", source_id)
