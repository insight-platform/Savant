"""Stall detector element for live source adapters."""

import inspect
import os
import sys
from enum import Enum
from threading import Event, Thread
from time import monotonic
from typing import Any, Optional

from savant.gstreamer import GObject, Gst
from savant.gstreamer.utils import (
    RequiredPropertyError,
    gst_post_library_settings_error,
    gst_post_stream_failed_error,
    gst_post_stream_failed_warning,
)
from savant.utils.log import LoggerMixin
from savant.utils.stall_detector import (
    StallEvaluator,
    StallSample,
    StallStatus,
    StallVerdict,
)

SINK_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink',
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.new_any(),
)
SRC_PAD_TEMPLATE = Gst.PadTemplate.new(
    'src',
    Gst.PadDirection.SRC,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.new_any(),
)


class StallAction(Enum):
    """What to do when a stall is detected."""

    NONE = 'none'
    """Only report the stall in the health file."""

    MESSAGE = 'message'
    """Log a warning and post a warning message on the bus."""

    FAIL = 'fail'
    """Post a bus error, ending the process non-zero so the supervisor restarts it."""


DEFAULT_MAX_IDLE_SECONDS = 30.0
DEFAULT_MIN_FPS = 0.0
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_CHECK_INTERVAL = 5.0
DEFAULT_WARMUP = 60.0
DEFAULT_STALL_ACTION = StallAction.MESSAGE

HEALTH_FILE_PERIOD_SECONDS = 5.0
"""Cap on the health file rewrite period, so a large check-interval cannot make
the container probe read a stale file."""

TERMINATION_GRACE_SECONDS = 10.0
"""How long `fail` waits for the posted error to end the process itself."""

STALL_EXIT_CODE = 75
"""Exit code used by the termination backstop."""

JOIN_TIMEOUT_SECONDS = 2.0
"""Bound on waiting for the evaluation thread when the element stops."""

ERROR_LOG_PERIOD_SECONDS = 60.0
"""Rate limit for logging failures of the evaluation tick."""

BLOCKED_LOG_PERIOD_SECONDS = 60.0
"""Rate limit for the blocked warning, which repeats for as long as the consumer
downstream stays stopped."""


class StallDetector(LoggerMixin, Gst.Element):
    """Detect a stalled live source and export a health signal. Live sources only:
    a fully read file, or a source quiet only from a blocked push, is not a stall.
    """

    GST_PLUGIN_NAME = 'stall_detector'

    __gstmetadata__ = (
        'Detect a stalled live source',
        'Transform',
        'Passes buffers through and detects when a live source stops delivering them.',
        'Oleg Abramov <abramov_ov@bitworks-software.online>',
    )

    __gsttemplates__ = (
        SINK_PAD_TEMPLATE,
        SRC_PAD_TEMPLATE,
    )

    __gproperties__ = {
        'max-idle-seconds': (
            float,
            'Max time without buffers',
            'Time without incoming buffers, in seconds, after which the source '
            'is considered stalled. Also the time after which an outstanding '
            'downstream push is considered blocked.',
            0,
            GObject.G_MAXDOUBLE,
            DEFAULT_MAX_IDLE_SECONDS,
            GObject.ParamFlags.READWRITE,
        ),
        'min-fps': (
            float,
            'Min acceptable buffer rate',
            'Minimum acceptable buffer rate over "window-seconds". '
            '0 disables the rate check, leaving "max-idle-seconds" the only '
            'stall criterion.',
            0,
            GObject.G_MAXDOUBLE,
            DEFAULT_MIN_FPS,
            GObject.ParamFlags.READWRITE,
        ),
        'window-seconds': (
            float,
            'Buffer rate window',
            'Window the buffer rate is measured over, in seconds. '
            'Ignored when "min-fps" is 0.',
            0,
            GObject.G_MAXDOUBLE,
            DEFAULT_WINDOW_SECONDS,
            GObject.ParamFlags.READWRITE,
        ),
        'check-interval': (
            float,
            'Evaluation interval',
            'Interval between stall evaluations, in seconds. When a health '
            f'file is configured, evaluations happen at least every '
            f'{HEALTH_FILE_PERIOD_SECONDS:.0f} seconds to keep the file fresh.',
            0,
            GObject.G_MAXDOUBLE,
            DEFAULT_CHECK_INTERVAL,
            GObject.ParamFlags.READWRITE,
        ),
        'warmup': (
            float,
            'Warmup period',
            'Grace period, in seconds, counted from the first time the element '
            'reaches PLAYING, during which no verdict is made.',
            0,
            GObject.G_MAXDOUBLE,
            DEFAULT_WARMUP,
            GObject.ParamFlags.READWRITE,
        ),
        'stall-action': (
            str,
            'Action on stall',
            'What to do when a stall is detected. One of '
            f'{", ".join(x.value for x in StallAction)}. "fail" posts an error '
            'on the bus, which ends the process with a non-zero status so that '
            'the container supervisor restarts it.',
            DEFAULT_STALL_ACTION.value,
            GObject.ParamFlags.READWRITE,
        ),
        'health-filepath': (
            str,
            'Health file path',
            'Path of the heartbeat file the detector rewrites with the current '
            'status. Unset disables the health file.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'probe-name': (
            str,
            'Name of the probe',
            'Name used to identify this detector in logs and in the health '
            'file. Defaults to the element name.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()

        # properties
        self.max_idle_seconds = DEFAULT_MAX_IDLE_SECONDS
        self.min_fps = DEFAULT_MIN_FPS
        self.window_seconds = DEFAULT_WINDOW_SECONDS
        self.check_interval = DEFAULT_CHECK_INTERVAL
        self.warmup = DEFAULT_WARMUP
        # Raw, validated at start: pygobject swallows a raise in do_set_property,
        # which would silently leave a safety element on its default.
        self.stall_action_name = DEFAULT_STALL_ACTION.value
        self.health_filepath: Optional[str] = None
        self.probe_name: Optional[str] = None

        self.stall_action = DEFAULT_STALL_ACTION

        # Streaming thread writes, evaluation thread reads. One writer per
        # attribute, one immutable read/write each, so no lock is needed.
        self._frames = 0
        self._last_arrival: Optional[float] = None
        self._push_started: Optional[float] = None
        self._epoch = 0
        self._eos = False

        self._warmup_origin: Optional[float] = None
        self._action_fired = False
        self._last_status: Optional[StallStatus] = None
        self._last_blocked_log: Optional[float] = None
        self._generation = 0
        self._stop_event: Optional[Event] = None
        self._thread: Optional[Thread] = None
        self._last_error_log: Optional[float] = None

        self.sink_pad: Gst.Pad = Gst.Pad.new_from_template(SINK_PAD_TEMPLATE, 'sink')
        self.src_pad: Gst.Pad = Gst.Pad.new_from_template(SRC_PAD_TEMPLATE, 'src')

        self.add_pad(self.sink_pad)
        self.add_pad(self.src_pad)

        self.sink_pad.set_chain_function_full(self.handle_buffer)
        self.sink_pad.add_probe(
            # EVENT_DOWNSTREAM does not cover flush events, and FLUSH_STOP re-arms.
            Gst.PadProbeType.EVENT_DOWNSTREAM | Gst.PadProbeType.EVENT_FLUSH,
            self.on_pad_event,
        )

    def do_get_property(self, prop: GObject.GParamSpec):
        """Gst plugin get property function."""
        if prop.name == 'max-idle-seconds':
            return self.max_idle_seconds
        if prop.name == 'min-fps':
            return self.min_fps
        if prop.name == 'window-seconds':
            return self.window_seconds
        if prop.name == 'check-interval':
            return self.check_interval
        if prop.name == 'warmup':
            return self.warmup
        if prop.name == 'stall-action':
            return self.stall_action_name
        if prop.name == 'health-filepath':
            return self.health_filepath
        if prop.name == 'probe-name':
            return self.probe_name
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function."""
        if prop.name == 'max-idle-seconds':
            self.max_idle_seconds = value
        elif prop.name == 'min-fps':
            self.min_fps = value
        elif prop.name == 'window-seconds':
            self.window_seconds = value
        elif prop.name == 'check-interval':
            self.check_interval = value
        elif prop.name == 'warmup':
            self.warmup = value
        elif prop.name == 'stall-action':
            self.stall_action_name = value
        elif prop.name == 'health-filepath':
            self.health_filepath = value
        elif prop.name == 'probe-name':
            self.probe_name = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_change_state(self, transition: Gst.StateChange) -> Gst.StateChangeReturn:
        """Validate the config; the only place a bad one can be refused, since
        `do_state_changed` runs after the state is already committed.
        """

        if transition == Gst.StateChange.NULL_TO_READY:
            try:
                self.validate_config()
            except RequiredPropertyError as exc:
                self.logger.exception('Failed to start element: %s', exc, exc_info=True)
                frame = inspect.currentframe()
                gst_post_library_settings_error(self, frame, __file__, text=exc.args[0])
                return Gst.StateChangeReturn.FAILURE

        # Cannot use `super()` since it is `self`
        return Gst.Element.do_change_state(self, transition)

    def do_state_changed(self, old: Gst.State, new: Gst.State, pending: Gst.State):
        """Start the thread when leaving NULL, arm the warmup on the first PLAYING,
        stop the thread on the way back to NULL.
        """

        if old == Gst.State.NULL and new != Gst.State.NULL:
            self.start_evaluation()

        # Only the first PLAYING arms the warmup, so that PAUSED <-> PLAYING
        # flapping cannot push the deadline out indefinitely.
        if new == Gst.State.PLAYING and self._warmup_origin is None:
            self._warmup_origin = monotonic()
            self.logger.info(
                '%s: armed, no verdict for the next %.2f seconds.',
                self.probe_name or self.get_name(),
                self.warmup,
            )

        if new == Gst.State.NULL:
            self.stop_evaluation()

    def start_evaluation(self):
        """Start the evaluation thread; config already validated in do_change_state."""

        if self._thread is not None and self._thread.is_alive():
            return

        self._generation += 1
        self._action_fired = False
        self._last_status = None
        self._last_blocked_log = None
        # A Thread cannot be restarted and READY <-> PAUSED cycles happen.
        self._stop_event = Event()
        self._thread = Thread(
            target=self.evaluation_job,
            args=(self._generation, self._stop_event),
            daemon=True,
        )
        self._thread.start()

    def stop_evaluation(self):
        """Stop the evaluation thread."""

        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(JOIN_TIMEOUT_SECONDS)
            self._thread = None

    def validate_config(self):
        """Check the properties and the health file, raise on a bad value."""

        if self.probe_name is None:
            self.probe_name = self.get_name()

        action_names = [x.value for x in StallAction]
        if self.stall_action_name not in action_names:
            raise RequiredPropertyError(
                f'"stall-action" must be one of {", ".join(action_names)}, '
                f'got "{self.stall_action_name}"'
            )
        self.stall_action = StallAction(self.stall_action_name)

        if self.max_idle_seconds <= 0:
            raise RequiredPropertyError('"max-idle-seconds" must be greater than 0')

        if self.check_interval <= 0:
            raise RequiredPropertyError('"check-interval" must be greater than 0')

        if self.min_fps > 0 and self.check_interval >= self.window_seconds:
            raise RequiredPropertyError(
                '"check-interval" must be less than "window-seconds" '
                'when "min-fps" is set'
            )

        if self.health_filepath is not None:
            try:
                self.write_health_file(StallStatus.STARTING, None, 0, None)
            except OSError as exc:
                raise RequiredPropertyError(
                    f'"health-filepath" {self.health_filepath} is not writable: {exc}'
                ) from exc

    def handle_buffer(
        self,
        sink_pad: Gst.Pad,
        element: Gst.Element,
        buffer: Gst.Buffer,
    ) -> Gst.FlowReturn:
        """Count the buffer, push it downstream and time the push."""

        now = monotonic()
        self._frames += 1
        self._last_arrival = now
        # Timed so the evaluator can tell a quiet source from a blocked push.
        self._push_started = now
        try:
            return self.src_pad.push(buffer)
        finally:
            # A push that raises must not leave the stream reading as blocked,
            # the one status that never fires an action.
            self._push_started = None

    def on_pad_event(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        """Handle sink pad event."""

        event: Gst.Event = info.get_event()
        self.logger.debug('Received event %s from %s', event.type, pad.get_name())
        if event.type == Gst.EventType.EOS:
            self.logger.info('Got EOS from %s', pad.get_name())
            self._eos = True

        elif event.type in (Gst.EventType.FLUSH_STOP, Gst.EventType.STREAM_START):
            # Restarted: history is stale, and a seeking source must be re-armed.
            self._epoch += 1
            self._eos = False

        elif event.type == Gst.EventType.CAPS:
            # Caps are the one event the default handler does not forward, since
            # that needs the PROXY_CAPS pad flag and pad flags are not settable
            # from Python. Everything else is left to the default handler:
            # forwarding it here as well delivers non-sticky events twice.
            self.src_pad.push_event(event)

        return Gst.PadProbeReturn.OK

    def evaluation_job(self, generation: int, stop_event: Event):
        """Evaluate the stream periodically until the element stops."""

        evaluator = StallEvaluator(
            max_idle_seconds=self.max_idle_seconds,
            min_fps=self.min_fps,
            window_seconds=self.window_seconds,
        )
        period = self.check_interval
        if self.health_filepath is not None:
            period = min(period, HEALTH_FILE_PERIOD_SECONDS)

        while not stop_event.wait(period):
            # A thread that outlived its join must never act.
            if generation != self._generation:
                return
            try:
                verdict = evaluator.evaluate(self.take_sample())
                self.handle_verdict(verdict, generation, stop_event)
            except Exception as exc:
                # An unwritable health file must not end stall detection.
                self.log_tick_error(exc)

    def take_sample(self) -> StallSample:
        """Read what the streaming thread measured."""

        now = monotonic()
        armed = (
            self._warmup_origin is not None and now - self._warmup_origin > self.warmup
        )

        return StallSample(
            time=now,
            frames=self._frames,
            last_arrival=self._last_arrival,
            push_started=self._push_started,
            epoch=self._epoch,
            armed=armed,
            eos=self._eos,
        )

    def handle_verdict(self, verdict: StallVerdict, generation: int, stop_event: Event):
        """Report the verdict and fire the action if the stream stalled."""

        self.write_health_file(
            verdict.status,
            verdict.fps,
            verdict.frames,
            verdict.seconds_since_last_frame,
        )
        previous_status = self._last_status
        self._last_status = verdict.status

        if verdict.status == StallStatus.BLOCKED:
            self.log_blocked(verdict, previous_status)
            return

        if verdict.status == StallStatus.RUNNING:
            # Re-arm, so a stall that follows a recovery is reported again.
            self._action_fired = False

        if verdict.status != StallStatus.STALLED:
            self.logger.debug(
                '%s: %s, %s frames.',
                self.probe_name,
                verdict.status.value,
                verdict.frames,
            )
            return

        if self._action_fired:
            return
        self._action_fired = True

        text = f'Source "{self.probe_name}" stalled. {verdict.reason}'
        if self.stall_action == StallAction.NONE:
            # "none" reports through the health file only.
            self.logger.debug(text)
            return

        self.logger.warning(text)
        frame = inspect.currentframe()
        gst_post_stream_failed_warning(
            gst_element=self, frame=frame, file_path=__file__, text=text
        )
        if self.stall_action != StallAction.FAIL:
            return

        self.logger.warning(
            '%s: terminating the pipeline so the supervisor restarts it.',
            self.probe_name,
        )
        gst_post_stream_failed_error(
            gst_element=self, frame=frame, file_path=__file__, text=text
        )
        self.terminate(verdict, generation, stop_event)

    def terminate(self, verdict: StallVerdict, generation: int, stop_event: Event):
        """Bound the teardown after the error is posted: the error normally stops
        this thread, but a wedged teardown would hang forever.
        """

        if stop_event.wait(TERMINATION_GRACE_SECONDS):
            return
        if generation != self._generation:
            return

        self.logger.error(
            '%s: pipeline did not stop within %.2f seconds after the error, '
            'exiting with code %s.',
            self.probe_name,
            TERMINATION_GRACE_SECONDS,
            STALL_EXIT_CODE,
        )
        try:
            self.write_health_file(
                verdict.status,
                verdict.fps,
                verdict.frames,
                verdict.seconds_since_last_frame,
            )
        except OSError:
            pass
        # fps_meter prints to stdout, which is block-buffered when not a tty.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        os._exit(STALL_EXIT_CODE)

    def write_health_file(
        self,
        status: StallStatus,
        fps: Optional[float],
        frames: int,
        seconds_since_last_frame: Optional[float],
    ):
        """Rewrite the health file atomically: replaced, not truncated, since the
        probe could otherwise observe a write in flight.
        """

        if self.health_filepath is None:
            return

        body = '\n'.join(
            [
                status.value,
                f'fps={_format_float(fps)}',
                f'frames={frames}',
                f'seconds-since-last-frame={_format_float(seconds_since_last_frame)}',
                f'probe={self.probe_name}',
                f'pid={os.getpid()}',
                '',
            ]
        )
        tmp_filepath = f'{self.health_filepath}.{os.getpid()}.tmp'
        with open(tmp_filepath, 'w', encoding='utf-8') as health_file:
            health_file.write(body)
        os.replace(tmp_filepath, self.health_filepath)

    def log_blocked(
        self, verdict: StallVerdict, previous_status: Optional[StallStatus]
    ):
        """Warn about a blocked push when it starts and periodically after, since
        the block lasts for as long as the consumer downstream stays stopped.
        """

        now = monotonic()
        started = previous_status != StallStatus.BLOCKED
        due = (
            self._last_blocked_log is None
            or now - self._last_blocked_log >= BLOCKED_LOG_PERIOD_SECONDS
        )
        if started or due:
            self._last_blocked_log = now
            self.logger.warning('%s: %s', self.probe_name, verdict.reason)
        else:
            self.logger.debug('%s: %s', self.probe_name, verdict.reason)

    def log_tick_error(self, exc: Exception):
        """Log a failed evaluation tick, at most once per period."""

        now = monotonic()
        if (
            self._last_error_log is not None
            and now - self._last_error_log < ERROR_LOG_PERIOD_SECONDS
        ):
            return
        self._last_error_log = now
        self.logger.error(
            '%s: stall evaluation tick failed: %s', self.probe_name, exc, exc_info=True
        )


def _format_float(value: Optional[float]) -> str:
    """Format a measurement for the health file."""

    return 'n/a' if value is None else f'{value:.2f}'


# register plugin
GObject.type_register(StallDetector)
__gstelementfactory__ = (
    StallDetector.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    StallDetector,
)
