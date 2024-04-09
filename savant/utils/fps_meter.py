"""FPS measurement."""

from time import time
from typing import Optional


# pylint:disable=too-many-instance-attributes
class FPSMeter:
    """FPS measurement.

    :param period_frames: FPS measurement period, in frames
    :param period_seconds: FPS measurement period, in seconds
    """

    def __init__(
        self,
        period_frames: Optional[int] = None,
        period_seconds: Optional[float] = None,
    ):
        self._period_frames = 100000
        self._period_seconds = None

        if period_frames is not None:
            assert (
                period_seconds is None
            ), 'Only one of "period_frames" or "period_seconds" should be set'
            self.period_frames = period_frames
        elif period_seconds is not None:
            self.period_seconds = period_seconds

        self._frame_counter = 0
        self._last_frame_counter = 0
        self._start_time = 0.0
        self._last_start_time = 0.0

    @property
    def period_frames(self):
        """FPS measurement period, in frames."""
        return self._period_frames

    @period_frames.setter
    def period_frames(self, value: int):
        """FPS measurement period, in frames."""
        assert value > 0
        self._period_frames = value
        self._period_seconds = None

    @property
    def period_seconds(self):
        """FPS measurement period, in seconds."""
        return self._period_seconds

    @period_seconds.setter
    def period_seconds(self, value: float):
        """FPS measurement period, in seconds."""
        assert value > 0
        self._period_frames = None
        self._period_seconds = value

    @property
    def frame_counter(self):
        """Frame number."""
        return self._last_frame_counter

    @property
    def exec_seconds(self):
        """Execution time, sec."""
        return time() - self._last_start_time

    @property
    def fps(self):
        """FPS."""
        return self.frame_counter / self.exec_seconds

    @property
    def message(self):
        """Default FPS measurement message."""
        frame_counter = self.frame_counter
        frame_word = 'frame' if frame_counter == 1 else 'frames'
        return f'Processed {frame_counter} {frame_word}, {self.fps:.2f} FPS.'

    def start(self):
        """Starts the meter."""
        self._frame_counter = 0
        self._last_frame_counter = 0
        self._start_time = time()
        self._last_start_time = 0.0

    def __call__(self, frame_count: int = 1) -> bool:
        """
        :return: True if FPS measurement has occurred
        """
        # first call: init
        if not self._start_time:
            self._start_time = time()
            return False

        # next call: increment
        self._frame_counter += frame_count
        self._last_frame_counter = self._frame_counter
        self._last_start_time = self._start_time

        # reset after period
        if self._period_passed():
            self.reset_counter()
            return True

        return False

    def reset_counter(self):
        """Reset frame counter."""
        self._frame_counter = 0
        self._start_time = time()

    def _period_passed(self):
        if self._period_seconds:
            return self.exec_seconds >= self._period_seconds

        return self._frame_counter >= self._period_frames
