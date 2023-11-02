from datetime import datetime

import cv2
import numpy as np

from adapters.ds.sinks.always_on_rtsp.utils import Frame
from savant.utils.logging import get_logger


class TimestampOverlay:
    def __init__(self):
        self.logger = get_logger(f'{self.__module__}.{self.__class__.__name__}')

        # TODO: make properties configurable
        self._width = 520
        self._height = 50
        self._location = (10, 35)
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1
        self._color = (255, 255, 255, 255)  # white
        self._thickness = 2
        self._line_type = cv2.LINE_AA

        self._placeholder = np.zeros((self._height, self._width, 4), dtype=np.uint8)
        self.logger.debug(
            'Timestamp placeholder size is %sx%s', self._width, self._height
        )

    def overlay_timestamp(
        self,
        width: int,
        height: int,
        frame: Frame,
        timestamp: datetime,
    ):
        # TODO: make timestamp precision configurable
        self.logger.debug(
            'Placing timestamp %s on a frame of a size %sx%s',
            timestamp,
            width,
            height,
        )
        if isinstance(frame, cv2.cuda.GpuMat):
            placeholder = self._placeholder
        else:
            placeholder = frame[0 : self._height, width - self._width :]
        placeholder.fill(0)
        cv2.putText(
            placeholder,
            str(timestamp),
            self._location,
            fontFace=self._font,
            fontScale=self._font_scale,
            color=self._color,
            thickness=self._thickness,
            lineType=self._line_type,
        )
        if isinstance(frame, cv2.cuda.GpuMat):
            cv2.cuda.GpuMat(
                frame,
                (
                    width - self._width,
                    0,
                    self._width,
                    self._height,
                ),
            ).upload(placeholder)
