"""Tests for the stall detector logic that needs no GStreamer pipeline."""

import os
import subprocess
import time
from pathlib import Path

import pytest

from savant.utils.stall_detector import StallEvaluator, StallSample, StallStatus

PROBE_SCRIPT = (
    Path(__file__).parent.parent / 'adapters' / 'shared' / 'pipeline_healthcheck.sh'
)


def sample(
    at: float,
    frames: int = 0,
    last_arrival=None,
    epoch_started=None,
    push_started=None,
    epoch: int = 0,
    armed: bool = True,
    eos: bool = False,
) -> StallSample:
    """Build a sample, defaulting to an armed stream with no history."""

    return StallSample(
        time=at,
        frames=frames,
        last_arrival=last_arrival,
        epoch_started=epoch_started,
        push_started=push_started,
        epoch=epoch,
        armed=armed,
        eos=eos,
    )


def feed(evaluator: StallEvaluator, fps: float, until: float, step: float = 5.0):
    """Feed a steady stream at ``fps`` and return the last verdict."""

    verdict = None
    at = step
    while at <= until:
        verdict = evaluator.evaluate(
            sample(at, frames=round(at * fps), last_arrival=at)
        )
        at += step
    return verdict


class TestStatus:
    def test_running_while_buffers_arrive(self):
        evaluator = StallEvaluator(max_idle_seconds=10)
        verdict = evaluator.evaluate(sample(5, frames=100, last_arrival=4))
        assert verdict.status is StallStatus.RUNNING
        assert verdict.reason is None

    def test_starting_before_armed(self):
        evaluator = StallEvaluator(max_idle_seconds=10)
        verdict = evaluator.evaluate(sample(100, armed=False))
        assert verdict.status is StallStatus.STARTING

    def test_stalled_when_idle_exceeds_max(self):
        evaluator = StallEvaluator(max_idle_seconds=10)
        assert (
            evaluator.evaluate(sample(10, frames=1, last_arrival=0)).status
            is StallStatus.RUNNING
        )
        verdict = evaluator.evaluate(sample(11, frames=1, last_arrival=0))
        assert verdict.status is StallStatus.STALLED
        assert '11.00 seconds' in verdict.reason

    def test_stalled_when_no_buffer_ever_arrived(self):
        evaluator = StallEvaluator(max_idle_seconds=10)
        verdict = evaluator.evaluate(sample(1, frames=0, last_arrival=None))
        assert verdict.status is StallStatus.STALLED
        assert 'since the element started' in verdict.reason

    def test_first_buffer_gets_the_full_idle_allowance(self):
        """Until the first buffer arrives the stream start is the idle baseline,
        so a warmup shorter than max-idle-seconds cannot stall the stream.
        """

        evaluator = StallEvaluator(max_idle_seconds=30)
        verdict = evaluator.evaluate(sample(20, frames=0, epoch_started=0))
        assert verdict.status is StallStatus.RUNNING
        assert verdict.seconds_since_last_frame == pytest.approx(20.0)

        verdict = evaluator.evaluate(sample(31, frames=0, epoch_started=0))
        assert verdict.status is StallStatus.STALLED
        assert '31.00 seconds' in verdict.reason

    def test_ended_after_eos(self):
        """EOS is not a stall, and it is not "starting" either: nothing to judge."""

        evaluator = StallEvaluator(max_idle_seconds=10)
        verdict = evaluator.evaluate(sample(100, frames=1, last_arrival=1, eos=True))
        assert verdict.status is StallStatus.ENDED
        assert 'ended with EOS' in verdict.reason

    def test_ended_takes_precedence_over_warmup(self):
        evaluator = StallEvaluator(max_idle_seconds=10)
        verdict = evaluator.evaluate(sample(100, armed=False, eos=True))
        assert verdict.status is StallStatus.ENDED

    def test_idle_exactly_at_threshold_is_running(self):
        """The threshold is exclusive, so a feed at exactly the limit is fine."""

        evaluator = StallEvaluator(max_idle_seconds=10)
        verdict = evaluator.evaluate(sample(10, frames=1, last_arrival=0))
        assert verdict.status is StallStatus.RUNNING


class TestBlocked:
    """A blocked downstream push must never be reported as a stall."""

    def test_blocked_when_push_outstanding_too_long(self):
        evaluator = StallEvaluator(max_idle_seconds=10)
        verdict = evaluator.evaluate(
            sample(30, frames=1, last_arrival=5, push_started=5)
        )
        assert verdict.status is StallStatus.BLOCKED
        assert 'held up downstream' in verdict.reason

    def test_blocked_takes_precedence_over_idle_stall(self):
        evaluator = StallEvaluator(max_idle_seconds=10)
        idle = evaluator.evaluate(sample(30, frames=1, last_arrival=5))
        blocked = evaluator.evaluate(
            sample(31, frames=1, last_arrival=5, push_started=5)
        )
        assert idle.status is StallStatus.STALLED
        assert blocked.status is StallStatus.BLOCKED

    def test_blocked_takes_precedence_over_warmup(self):
        evaluator = StallEvaluator(max_idle_seconds=10)
        verdict = evaluator.evaluate(
            sample(30, frames=1, last_arrival=5, push_started=5, armed=False)
        )
        assert verdict.status is StallStatus.BLOCKED

    def test_short_push_is_not_blocked(self):
        evaluator = StallEvaluator(max_idle_seconds=10)
        verdict = evaluator.evaluate(
            sample(30, frames=1, last_arrival=29.5, push_started=29.5)
        )
        assert verdict.status is StallStatus.RUNNING


class TestRateCheck:
    def test_min_fps_zero_disables_rate_check(self):
        """One buffer every 10s is fine when only the idle timeout is active."""

        evaluator = StallEvaluator(max_idle_seconds=30, min_fps=0, window_seconds=30)
        verdict = feed(evaluator, fps=0.1, until=100, step=10)
        assert verdict.status is StallStatus.RUNNING

    def test_degraded_rate_is_stalled(self):
        evaluator = StallEvaluator(max_idle_seconds=30, min_fps=5, window_seconds=30)
        verdict = feed(evaluator, fps=2, until=60)
        assert verdict.status is StallStatus.STALLED
        assert 'below min-fps' in verdict.reason

    def test_healthy_rate_is_running(self):
        evaluator = StallEvaluator(max_idle_seconds=30, min_fps=5, window_seconds=30)
        verdict = feed(evaluator, fps=10, until=60)
        assert verdict.status is StallStatus.RUNNING

    def test_rate_arm_waits_for_a_full_window(self):
        """A partially filled window must not fire, however low the rate is."""

        evaluator = StallEvaluator(max_idle_seconds=30, min_fps=5, window_seconds=30)
        for at in (5, 10, 15, 20, 25):
            verdict = evaluator.evaluate(sample(at, frames=0, last_arrival=at))
            assert verdict.status is StallStatus.RUNNING, f'fired at t={at}'
        assert evaluator.evaluate(sample(35, frames=0, last_arrival=35)).status is (
            StallStatus.STALLED
        )

    def test_rate_uses_covered_span_not_nominal_window(self):
        """Dividing by the window would halve the rate while the window fills."""

        evaluator = StallEvaluator(max_idle_seconds=30, min_fps=5, window_seconds=30)
        at = 5.0
        while at <= 60:
            verdict = evaluator.evaluate(
                sample(at, frames=round(at * 10), last_arrival=at)
            )
            if verdict.fps is not None:
                assert verdict.fps == pytest.approx(10.0), f'wrong rate at t={at}'
            at += 5

    def test_fps_is_none_on_the_first_sample_only(self):
        evaluator = StallEvaluator(max_idle_seconds=30, window_seconds=30)
        assert evaluator.evaluate(sample(5, frames=50, last_arrival=5)).fps is None
        assert evaluator.evaluate(sample(10, frames=100, last_arrival=10)).fps == (
            pytest.approx(10.0)
        )


class TestWindow:
    def test_epoch_bump_clears_history(self):
        evaluator = StallEvaluator(max_idle_seconds=30, min_fps=5, window_seconds=30)
        feed(evaluator, fps=10, until=60)
        verdict = evaluator.evaluate(sample(65, frames=0, last_arrival=65, epoch=1))
        assert verdict.fps is None

    def test_rate_cannot_fire_on_a_refilling_window(self):
        evaluator = StallEvaluator(max_idle_seconds=30, min_fps=5, window_seconds=30)
        feed(evaluator, fps=10, until=60)
        # Frame count restarts from 0, so the rate reads 0 without a full window.
        for at in (65, 70, 75):
            verdict = evaluator.evaluate(sample(at, frames=0, last_arrival=at, epoch=1))
            assert verdict.status is StallStatus.RUNNING, f'fired at t={at}'

    def test_span_covers_a_full_window_and_deque_stays_bounded(self):
        evaluator = StallEvaluator(max_idle_seconds=30, window_seconds=30)
        feed(evaluator, fps=10, until=300)
        assert evaluator._span() >= 30
        # window / step + 2 is the bound: one extra sample outside the window.
        assert len(evaluator._samples) <= 30 / 5 + 2


class TestVerdictFields:
    def test_measurements_are_reported(self):
        evaluator = StallEvaluator(max_idle_seconds=30, window_seconds=30)
        evaluator.evaluate(sample(5, frames=50, last_arrival=5))
        verdict = evaluator.evaluate(sample(10, frames=100, last_arrival=9))
        assert verdict.frames == 100
        assert verdict.fps == pytest.approx(10.0)
        assert verdict.seconds_since_last_frame == pytest.approx(1.0)

    def test_seconds_since_last_frame_is_none_before_the_stream_starts(self):
        evaluator = StallEvaluator(max_idle_seconds=30)
        verdict = evaluator.evaluate(sample(5, frames=0, last_arrival=None))
        assert verdict.seconds_since_last_frame is None


@pytest.mark.skipif(not PROBE_SCRIPT.exists(), reason='probe script not found')
class TestPipelineHealthcheck:
    """The container probe must fail closed on anything but a fresh good status."""

    @staticmethod
    def probe(health_filepath: Path, **env) -> int:
        result = subprocess.run(
            ['sh', str(PROBE_SCRIPT)],
            env={
                'PATH': os.environ['PATH'],
                'PIPELINE_HEALTH_FILEPATH': str(health_filepath),
                **env,
            },
            capture_output=True,
        )
        return result.returncode

    def test_missing_file_is_healthy(self, tmp_path):
        """Absence is how the probe knows the detector is not enabled here."""

        assert self.probe(tmp_path / 'absent.txt') == 0

    @pytest.mark.parametrize('status', ['running', 'starting'])
    def test_good_status_is_healthy(self, tmp_path, status):
        health_filepath = tmp_path / 'health.txt'
        health_filepath.write_text(f'{status}\nfps=24.97\nframes=100\n')
        assert self.probe(health_filepath) == 0

    @pytest.mark.parametrize(
        'body',
        [
            'stalled\nfps=0.00\n',
            'blocked\n',
            'ended\n',
            '',
            'runningX\n',
            'garbage\n',
            # Would pass the `grep -q running` idiom used by the module probe.
            'stalled\nprobe=running-cam\n',
        ],
    )
    def test_bad_status_is_unhealthy(self, tmp_path, body):
        health_filepath = tmp_path / 'health.txt'
        health_filepath.write_text(body)
        assert self.probe(health_filepath) == 1

    def test_malformed_max_age_is_unhealthy(self, tmp_path):
        """A bad max age must fail closed, not skip the freshness check."""

        health_filepath = tmp_path / 'health.txt'
        health_filepath.write_text('running\n')
        assert self.probe(health_filepath, PIPELINE_HEALTH_MAX_AGE='soon') == 1
        assert self.probe(health_filepath, PIPELINE_HEALTH_MAX_AGE='30s') == 1
        # An empty value is how the shell spells "unset", so it takes the default.
        assert self.probe(health_filepath, PIPELINE_HEALTH_MAX_AGE='') == 0

    def test_stale_file_is_unhealthy(self, tmp_path):
        health_filepath = tmp_path / 'health.txt'
        health_filepath.write_text('running\n')
        old = time.time() - 60
        os.utime(health_filepath, (old, old))
        assert self.probe(health_filepath) == 1
        assert self.probe(health_filepath, PIPELINE_HEALTH_MAX_AGE='120') == 0
