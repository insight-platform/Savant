from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
from savant_rs.primitives import VideoFrame


class BaseIngressConverter(ABC):
    @abstractmethod
    def convert(
        self,
        source_id: str,
        meta: VideoFrame,
        frame: cv2.cuda.GpuMat,
    ) -> cv2.cuda.GpuMat:
        pass

    @abstractmethod
    def on_stream_start(
        self,
        source_id: str,
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int]]:
        pass

    @abstractmethod
    def on_stream_stop(self, source_id: str) -> None:
        pass
