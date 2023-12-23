import random
import time
from typing import Iterator

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst


class LagGenerator(NvDsPyFuncPlugin):
    """Simulates load spikes by lagging frames for a random amount of time.

    :param frame_processing_time: Time to process a frame in seconds.
        Needed to limit module performance when no lag is emulated,
        so the buffer eventually would be full.
    :param min_lag: Minimum lag time in seconds.
    :param max_lag: Maximum lag time in seconds.
    :param lag_frames_interval: Lagged frames sequence length.
    :param pass_frames_interval: Non-lagged frames sequence length.
    """

    def __init__(
        self,
        frame_processing_time: float = 0.01,
        min_lag: float = 0.06,
        max_lag: float = 0.1,
        lag_frames_interval: int = 500,
        pass_frames_interval: int = 500,
        **kwargs,
    ):
        self.frame_processing_time = frame_processing_time
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.lag_frames_interval = lag_frames_interval
        self.pass_frames_interval = pass_frames_interval
        self.lag_generator = self.create_lag_generator()
        super().__init__(**kwargs)

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        lag = next(self.lag_generator)
        self.logger.debug(
            'Lag frame for source %s with PTS %s for %s seconds.',
            frame_meta.source_id,
            frame_meta.pts,
            lag,
        )
        time.sleep(lag + self.frame_processing_time)

    def create_lag_generator(self) -> Iterator[float]:
        """Generates lag values for frames."""

        while True:
            lag = random.uniform(self.min_lag, self.max_lag)
            self.logger.info(
                'Lagging %s frames for %s seconds.',
                self.lag_frames_interval,
                lag,
            )
            for _ in range(self.lag_frames_interval):
                yield lag
            self.logger.info('Passing %s frame without lag.', self.pass_frames_interval)
            for _ in range(self.pass_frames_interval):
                yield 0
