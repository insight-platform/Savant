from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
from savant_rs.primitives import VideoFrame

from savant.utils.logging import get_logger


class BaseIngressConverter(ABC):
    def __init__(self):
        self.logger = get_logger(self.__module__)

    @abstractmethod
    def convert(
        self,
        source_id: str,
        meta: VideoFrame,
        in_frame: 'cv2.cuda.GpuMat',
        out_frame: 'cv2.cuda.GpuMat',
    ):
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
