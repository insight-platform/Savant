"""SavantRsVideoDemux element."""
import inspect
import time
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Dict, Optional

from pygstsavantframemeta import gst_buffer_get_savant_frame_meta
from savant_rs.pipeline2 import VideoPipeline

from gst_plugins.python.savant_rs_video_demux_common import FrameParams, build_caps
from savant.gstreamer import GObject, Gst
from savant.gstreamer.codecs import Codec
from savant.gstreamer.event import parse_savant_eos_event
from savant.gstreamer.utils import (
    RequiredPropertyError,
    gst_post_library_settings_error,
    gst_post_stream_demux_error,
    on_pad_event,
    required_property,
)
from savant.utils.logging import LoggerMixin

DEFAULT_SOURCE_TIMEOUT = 60
DEFAULT_SOURCE_EVICTION_INTERVAL = 15
OUT_CAPS = Gst.Caps.from_string(';'.join(x.value.caps_with_params for x in Codec))

SAVANT_RS_VIDEO_DEMUX_PROPERTIES = {
    'source-timeout': (
        int,
        'Source timeout',
        'Timeout before deleting stale source (in seconds)',
        0,
        2147483647,
        DEFAULT_SOURCE_TIMEOUT,
        GObject.ParamFlags.READWRITE,
    ),
    'source-eviction-interval': (
        int,
        'Source eviction interval',
        'Interval between source evictions (in seconds)',
        0,
        2147483647,
        DEFAULT_SOURCE_EVICTION_INTERVAL,
        GObject.ParamFlags.READWRITE,
    ),
    'eos-on-timestamps-reset': (
        bool,
        'Send EOS when timestamps reset',
        'Send EOS when timestamps reset (i.e. non-monotonous). '
        'Needed to prevent decoder from changing PTS on "decreasing timestamp" error.',
        False,
        GObject.ParamFlags.READWRITE,
    ),
    'max-parallel-streams': (
        int,
        'Maximum number of parallel streams.',
        'Maximum number of parallel streams (0 = unlimited).',
        0,
        GObject.G_MAXINT,
        0,
        GObject.ParamFlags.READWRITE,
    ),
    'pipeline': (
        object,
        'VideoPipeline object from savant-rs.',
        'VideoPipeline object from savant-rs.',
        GObject.ParamFlags.READWRITE,
    ),
    'pipeline-stage-name': (
        str,
        'Name of the pipeline stage.',
        'Name of the pipeline stage.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
}


@dataclass
class SourceInfo:
    """Info about source/camera."""

    source_id: str
    params: FrameParams
    src_pad: Optional[Gst.Pad] = None
    timestamp: float = 0
    last_pts: int = 0
    last_dts: int = 0


class SavantRsVideoDemux(LoggerMixin, Gst.Element):
    """Deserializes savant-rs video stream and demultiplex them by source ID."""

    GST_PLUGIN_NAME = 'savant_rs_video_demux'

    __gstmetadata__ = (
        'Savant-rs video demuxer',
        'Demuxer',
        'Deserializes savant-rs video stream and demultiplex them by source ID. '
        'Outputs encoded video frames to src pad "src_<source_id>".',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'sink',
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        ),
        Gst.PadTemplate.new(
            'src_%s',
            Gst.PadDirection.SRC,
            Gst.PadPresence.SOMETIMES,
            OUT_CAPS,
        ),
    )

    __gproperties__ = SAVANT_RS_VIDEO_DEMUX_PROPERTIES

    __gsignals__ = {'shutdown': (GObject.SignalFlags.RUN_LAST, None, ())}

    def __init__(self):
        super().__init__()
        self.sources: Dict[str, SourceInfo] = {}
        self.eos_on_timestamps_reset = False
        self.source_timeout = DEFAULT_SOURCE_TIMEOUT
        self.source_eviction_interval = DEFAULT_SOURCE_EVICTION_INTERVAL
        self.last_eviction = 0
        self.source_lock = Lock()
        self.is_running = False
        self.expiration_thread = Thread(target=self.eviction_job, daemon=True)
        self.max_parallel_streams: int = 0
        self.video_pipeline: Optional[VideoPipeline] = None
        self.pipeline_stage_name: Optional[str] = None

        self.sink_pad: Gst.Pad = Gst.Pad.new_from_template(
            Gst.PadTemplate.new(
                'sink',
                Gst.PadDirection.SINK,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.new_any(),
            ),
            'sink',
        )
        self.sink_pad.set_chain_function(self.handle_buffer)
        self.sink_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {
                Gst.EventType.CUSTOM_DOWNSTREAM: self.on_savant_eos_event,
                Gst.EventType.EOS: self.on_eos,
            },
        )
        assert self.add_pad(self.sink_pad), 'Failed to add sink pad.'

    def do_state_changed(self, old: Gst.State, new: Gst.State, pending: Gst.State):
        """Process state change.
        Start an expiration thread if state changed from NULL.
        Init ingress filter if changed NULL -> READY
        """
        if (
            old == Gst.State.NULL
            and new != Gst.State.NULL
            and not self.expiration_thread.is_alive()
        ):
            self.is_running = True
            self.expiration_thread.start()

        if old == Gst.State.NULL and new == Gst.State.READY:
            try:
                required_property('pipeline', self.video_pipeline)
                required_property('pipeline-stage-name', self.pipeline_stage_name)
            except RequiredPropertyError as exc:
                self.logger.exception('Failed to start element: %s', exc, exc_info=True)
                frame = inspect.currentframe()
                gst_post_library_settings_error(self, frame, __file__, text=exc.args[0])
                return

    def do_get_property(self, prop):
        """Get property callback."""
        if prop.name == 'source-timeout':
            return self.source_timeout
        if prop.name == 'source-eviction-interval':
            return self.source_eviction_interval
        if prop.name == 'eos-on-timestamps-reset':
            return self.eos_on_timestamps_reset
        if prop.name == 'max-parallel-streams':
            return self.max_parallel_streams
        if prop.name == 'pipeline':
            return self.video_pipeline
        if prop.name == 'pipeline-stage-name':
            return self.pipeline_stage_name
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        """Set property callback."""
        if prop.name == 'source-timeout':
            self.source_timeout = value
        elif prop.name == 'source-eviction-interval':
            self.source_eviction_interval = value
        elif prop.name == 'eos-on-timestamps-reset':
            self.eos_on_timestamps_reset = value
        elif prop.name == 'max-parallel-streams':
            self.max_parallel_streams = value
        elif prop.name == 'pipeline':
            self.video_pipeline = value
        elif prop.name == 'pipeline-stage-name':
            self.pipeline_stage_name = value
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def handle_buffer(
        self,
        sink_pad: Gst.Pad,
        buffer: Gst.Buffer,
    ) -> Gst.FlowReturn:
        """Handle buffer from sink pad."""

        self.logger.debug(
            'Handling buffer of size %s with timestamp %s',
            buffer.get_size(),
            buffer.pts,
        )

        savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
        if savant_frame_meta is None:
            self.logger.warning(
                'No Savant Frame Metadata found on buffer with PTS %s, skipping.',
                buffer.pts,
            )
            return Gst.FlowReturn.OK

        frame_idx = savant_frame_meta.idx
        video_frame, _ = self.video_pipeline.get_independent_frame(frame_idx)
        self.video_pipeline.move_as_is(self.pipeline_stage_name, [frame_idx])
        frame_params = FrameParams.from_video_frame(video_frame)
        self.logger.debug(
            'Handling frame with PTS %s from source %s; frame %s a keyframe',
            buffer.pts,
            video_frame.source_id,
            'is' if video_frame.keyframe else 'is not',
        )
        if not self.is_running:
            self.logger.info(
                'Demuxer is not running. Skipping buffer with timestamp %s.',
                buffer.pts,
            )
            self.video_pipeline.delete(frame_idx)
            return Gst.FlowReturn.OK

        with self.source_lock:
            source_info: SourceInfo = self.sources.get(video_frame.source_id)
            if source_info is None:
                if (
                    self.max_parallel_streams
                    and len(self.sources) >= self.max_parallel_streams
                ):
                    self.is_running = False
                    error = (
                        f'Failed to add source {video_frame.source_id!r}: reached maximum '
                        f'number of streams: {self.max_parallel_streams}.'
                    )
                    self.logger.error(error)
                    frame = inspect.currentframe()
                    gst_post_stream_demux_error(
                        gst_element=self,
                        frame=frame,
                        file_path=__file__,
                        text=error,
                    )
                    self.video_pipeline.delete(frame_idx)
                    return Gst.FlowReturn.ERROR
                if not video_frame.keyframe:
                    self.logger.warning(
                        'Frame %s from source %s is not a keyframe, skipping it. '
                        'Stream should start with a keyframe.',
                        buffer.pts,
                        video_frame.source_id,
                    )
                    self.video_pipeline.delete(frame_idx)
                    return Gst.FlowReturn.OK
                source_info = SourceInfo(video_frame.source_id, frame_params)
                self.sources[video_frame.source_id] = source_info
            source_info.timestamp = time.time()
        if source_info.src_pad is not None and source_info.params != frame_params:
            self.update_frame_params(source_info, frame_params)
        if source_info.src_pad is not None:
            self.check_timestamps(source_info, buffer)
        source_info.last_pts = buffer.pts
        source_info.last_dts = buffer.dts
        if source_info.src_pad is None:
            if video_frame.keyframe:
                self.add_source(video_frame.source_id, source_info)
            else:
                self.logger.warning(
                    'Frame %s from source %s is not a keyframe, skipping it. '
                    'Stream should start with a keyframe.',
                    buffer.pts,
                    video_frame.source_id,
                )
                self.video_pipeline.delete(frame_idx)
                return Gst.FlowReturn.OK

        self.logger.debug(
            'Pushing frame with IDX %s and PTS %s from source %s',
            frame_idx,
            buffer.pts,
            video_frame.source_id,
        )
        result: Gst.FlowReturn = source_info.src_pad.push(buffer)

        self.logger.debug(
            'Frame with PTS %s from source %s has been processed.',
            buffer.pts,
            video_frame.source_id,
        )

        return result

    def on_savant_eos_event(
        self,
        sink_pad: Gst.Pad,
        event: Gst.Event,
    ) -> Gst.PadProbeReturn:
        """Handle savant-eos event from a sink pad."""

        self.logger.debug('Got CUSTOM_DOWNSTREAM event from %s', sink_pad.get_name())
        source_id = parse_savant_eos_event(event)
        if source_id is None:
            return Gst.PadProbeReturn.PASS

        self.logger.debug('Got savant-eos event for source %s', source_id)
        with self.source_lock:
            source_info: SourceInfo = self.sources.get(source_id)
            if source_info is None:
                return Gst.PadProbeReturn.DROP
            source_info.timestamp = time.time()

        if source_info.src_pad is not None:
            self.send_eos(source_info)
        del self.sources[source_id]

        return Gst.PadProbeReturn.DROP

    def on_eos(self, pad: Gst.Pad, event: Gst.Event) -> Gst.PadProbeReturn:
        """Handle EOS event from a sink pad.

        Emit shutdown signal.
        """

        self.logger.info('Received EOS.')
        with self.source_lock:
            self.is_running = False
            for source_id, source_info in list(self.sources.items()):
                self.logger.debug('Sending EOS to source %s.', source_id)
                if source_info.src_pad is not None:
                    self.send_eos(source_info)
                del self.sources[source_id]
        self.logger.debug('Emitting shutdown signal.')
        self.emit('shutdown')

        return Gst.FlowReturn.OK

    def add_source(self, source_id: str, source_info: SourceInfo):
        """Handle adding new source."""

        caps = build_caps(source_info.params)
        source_info.src_pad = Gst.Pad.new_from_template(
            Gst.PadTemplate.new(
                'src_%s',
                Gst.PadDirection.SRC,
                Gst.PadPresence.SOMETIMES,
                caps,
            ),
            f'src_{source_id}',
        )
        assert source_info.src_pad.set_active(True), 'Failed to set pad active.'
        assert self.add_pad(source_info.src_pad), 'Failed to add pad.'

        stream_id = source_info.src_pad.create_stream_id(
            self, self.sink_pad.get_stream_id()
        )
        self.logger.debug(
            'Starting new stream for source %s with stream id %s',
            source_id,
            stream_id,
        )
        source_info.src_pad.push_event(Gst.Event.new_stream_start(stream_id))

        segment: Gst.Segment = Gst.Segment.new()
        segment.init(Gst.Format.TIME)
        self.logger.debug('Starting new segment for source %s', source_id)
        source_info.src_pad.push_event(Gst.Event.new_segment(segment))

        self.logger.info(
            f'Created new src pad for source {source_id}: {source_info.src_pad.name}.'
        )

    def update_frame_params(self, source_info: SourceInfo, frame_params: FrameParams):
        """Handle changed frame parameters on a source."""

        if source_info.params != frame_params:
            self.logger.info(
                'Frame parameters on pad %s was changed from %s to %s',
                source_info.src_pad.get_name(),
                source_info.params,
                frame_params,
            )
            source_info.params = frame_params
            self.send_eos(source_info)
            return

        caps = build_caps(frame_params)
        source_info.src_pad.push_event(Gst.Event.new_caps(caps))
        self.logger.info(
            'Caps on pad %s changed to %s', source_info.src_pad, caps.to_string()
        )

    def check_timestamps(self, source_info: SourceInfo, buffer: Gst.Buffer):
        """Check frame timestamps (PTS and DTS).

        When timestamps are not monotonous, send EOS to prevent decoder
        from changing PTS on "decreasing timestamp" error.
        """

        if not self.eos_on_timestamps_reset:
            return
        self.logger.debug(
            'Timestamps on source %s updated. PTS: %s -> %s, DTS: %s -> %s',
            source_info.source_id,
            source_info.last_pts,
            buffer.pts,
            source_info.last_dts,
            buffer.dts,
        )
        reset = False
        if buffer.dts != Gst.CLOCK_TIME_NONE:
            reset = buffer.dts < source_info.last_dts
        if buffer.pts != Gst.CLOCK_TIME_NONE:
            if buffer.dts == Gst.CLOCK_TIME_NONE:
                reset = buffer.pts < source_info.last_pts

        if reset:
            self.logger.info(
                'Timestamps on source %s non-monotonous. Resetting source.',
                source_info.source_id,
            )
            self.send_eos(source_info)

    def send_eos(self, source_info: SourceInfo):
        """Send EOS event to a src pad."""
        self.logger.debug(
            'Sending EOS event to pad %s.',
            source_info.src_pad.get_name(),
        )
        source_info.src_pad.push_event(Gst.Event.new_eos())
        self.logger.debug('Removing pad %s', source_info.src_pad.get_name())
        self.remove_pad(source_info.src_pad)
        if self.video_pipeline is not None:
            try:
                self.video_pipeline.clear_source_ordering(source_info.source_id)
            except ValueError:
                pass
        source_info.src_pad = None

    def eviction_job(self):
        """Eviction job."""
        while self.is_running:
            self.eviction_loop()

    def eviction_loop(self):
        """Eviction job loop."""
        self.logger.debug('Start eviction loop')
        with self.source_lock:
            if not self.is_running:
                return
            now = time.time()
            limit = now - self.source_timeout
            earliest_ts = now
            for source_id, source_info in list(self.sources.items()):
                if source_info.timestamp < limit:
                    self.logger.debug('Source %s has expired', source_id)
                    if source_info.src_pad is not None:
                        self.send_eos(source_info)
                    del self.sources[source_id]
                else:
                    earliest_ts = min(earliest_ts, source_info.timestamp)
        wait = max(earliest_ts - limit, self.source_eviction_interval)
        self.logger.debug('Waiting %s seconds for the next eviction loop', wait)
        time.sleep(wait)


# register plugin
GObject.type_register(SavantRsVideoDemux)
__gstelementfactory__ = (
    SavantRsVideoDemux.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SavantRsVideoDemux,
)
