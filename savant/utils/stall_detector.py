"""Stall detection for live sources. Free of GStreamer imports, so the verdict
logic is testable without a pipeline.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional, Tuple


class StallStatus(Enum):
    """Status of the monitored stream."""

    STARTING = 'starting'
    """Not armed yet, no judgement made."""

    RUNNING = 'running'
    """Buffers are arriving as expected."""

    BLOCKED = 'blocked'
    """Quiet because of a downstream push: the fault is not upstream."""

    ENDED = 'ended'
    """The stream ended with EOS, so no verdict can be made any more."""

    STALLED = 'stalled'
    """Quiet with nothing downstream holding the thread."""


@dataclass
class StallSample:
    """Measurements taken by the streaming thread."""

    time: float
    frames: int
    last_arrival: Optional[float]
    # Start of the current epoch, the baseline for the idle timeout until the
    # first buffer arrives. None until the stream starts.
    epoch_started: Optional[float]
    # Start of the outstanding downstream push, None if no push is outstanding.
    push_started: Optional[float]
    # Bumped when the stream restarts, which invalidates the collected history.
    epoch: int
    # Warmup has passed.
    armed: bool
    # The stream ended with EOS.
    eos: bool = False


@dataclass
class StallVerdict:
    """Verdict for a single sample. ``reason`` is set for every status but RUNNING."""

    status: StallStatus
    frames: int
    fps: Optional[float]
    seconds_since_last_frame: Optional[float]
    reason: Optional[str] = None


class StallEvaluator:
    """Turns samples of a stream into stall verdicts. ``max_idle_seconds`` also
    marks an outstanding push as blocked; ``min_fps`` 0 disables the rate check.
    """

    def __init__(
        self,
        max_idle_seconds: float,
        min_fps: float = 0,
        window_seconds: float = 30,
    ):
        self.max_idle_seconds = max_idle_seconds
        self.min_fps = min_fps
        self.window_seconds = window_seconds

        # Private, so the streaming thread and the evaluator share no mutable state.
        self._samples: Deque[Tuple[float, int]] = deque()
        self._epoch: Optional[int] = None

    def evaluate(self, sample: StallSample) -> StallVerdict:
        """Add the sample to the window and return the verdict for it."""

        if sample.epoch != self._epoch:
            self._samples.clear()
            self._epoch = sample.epoch

        self._add_sample(sample.time, sample.frames)
        fps = self._fps()

        idle = None
        if sample.last_arrival is not None:
            idle = sample.time - sample.last_arrival
        elif sample.epoch_started is not None:
            # No buffer has arrived yet, so the first one gets the full idle
            # allowance, counted from the start of the stream. Counting from the
            # arrival that never happened would stall the stream on the first
            # armed sample whenever the warmup is shorter than the allowance.
            idle = sample.time - sample.epoch_started

        if (
            sample.push_started is not None
            and sample.time - sample.push_started > self.max_idle_seconds
        ):
            return StallVerdict(
                status=StallStatus.BLOCKED,
                frames=sample.frames,
                fps=fps,
                seconds_since_last_frame=idle,
                reason=(
                    f'Downstream push has been outstanding for '
                    f'{sample.time - sample.push_started:.2f} seconds, '
                    f'the stream is held up downstream.'
                ),
            )

        if sample.eos:
            return StallVerdict(
                status=StallStatus.ENDED,
                frames=sample.frames,
                fps=fps,
                seconds_since_last_frame=idle,
                reason='The stream ended with EOS.',
            )

        if not sample.armed:
            return StallVerdict(
                status=StallStatus.STARTING,
                frames=sample.frames,
                fps=fps,
                seconds_since_last_frame=idle,
                reason='Stall verdicts are not armed yet.',
            )

        if idle is None or idle > self.max_idle_seconds:
            if idle is None:
                reason = (
                    f'No buffer has arrived since the element started, '
                    f'max-idle-seconds is {self.max_idle_seconds:.2f}.'
                )
            else:
                reason = (
                    f'No buffer has arrived for {idle:.2f} seconds, '
                    f'max-idle-seconds is {self.max_idle_seconds:.2f}.'
                )
            return StallVerdict(
                status=StallStatus.STALLED,
                frames=sample.frames,
                fps=fps,
                seconds_since_last_frame=idle,
                reason=reason,
            )

        # The rate check needs a full window, since that is the interval min-fps
        # is specified over. The reported rate does not; it is informational.
        if (
            self.min_fps > 0
            and fps is not None
            and fps < self.min_fps
            and self._span() >= self.window_seconds
        ):
            return StallVerdict(
                status=StallStatus.STALLED,
                frames=sample.frames,
                fps=fps,
                seconds_since_last_frame=idle,
                reason=(
                    f'Buffer rate {fps:.2f} FPS is below min-fps '
                    f'{self.min_fps:.2f} over the last {self._span():.2f} seconds.'
                ),
            )

        return StallVerdict(
            status=StallStatus.RUNNING,
            frames=sample.frames,
            fps=fps,
            seconds_since_last_frame=idle,
        )

    def _add_sample(self, now: float, frames: int):
        """Append a sample and drop the ones the window no longer needs."""

        self._samples.append((now, frames))
        # Keep one sample outside the window, so the span can reach a full window.
        while (
            len(self._samples) > 2 and now - self._samples[1][0] >= self.window_seconds
        ):
            self._samples.popleft()

    def _span(self) -> float:
        """Time actually covered by the collected samples."""

        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    def _fps(self) -> Optional[float]:
        """Buffer rate over the covered span, None if not measurable. Dividing by
        the nominal window would halve the rate while the window fills.
        """

        span = self._span()
        if span <= 0:
            return None
        return (self._samples[-1][1] - self._samples[0][1]) / span
