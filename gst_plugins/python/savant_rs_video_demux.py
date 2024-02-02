"""SavantRsVideoDemux element."""
import inspect
import itertools
import time
from dataclasses import dataclass
from threading import Lock, Thread
from typing import Dict, Optional

from savant_rs.pipeline2 import VideoPipeline
from savant_rs.primitives import (
    EndOfStream,
    Shutdown,
    VideoFrame,
    VideoFrameContent,
    VideoFrameTransformation,
)
from savant_rs.utils import PropagatedContext

from gst_plugins.python.pyfunc_common import handle_non_fatal_error, init_pyfunc
from gst_plugins.python.savant_rs_video_demux_common import FrameParams, build_caps
from savant.api.enums import ExternalFrameType
from savant.api.parser import convert_ts
from savant.base.frame_filter import DefaultIngressFilter
from savant.base.pyfunc import PyFunc
from savant.gstreamer import GObject, Gst
from savant.gstreamer.codecs import Codec
from savant.gstreamer.utils import (
    gst_post_stream_demux_error,
    load_message_from_gst_buffer,
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
    'shutdown-auth': (
        str,
        'Authentication key for Shutdown message.',
        'Authentication key for Shutdown message.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'max-width': (
        int,
        'Maximum allowable resolution width of the video stream',
        'Maximum allowable resolution width of the video stream',
        0,
        GObject.G_MAXINT,
        0,
        GObject.ParamFlags.READWRITE,
    ),
    'max-height': (
        int,
        'Maximum allowable resolution height of the video stream',
        'Maximum allowable resolution height of the video stream',
        0,
        GObject.G_MAXINT,
        0,
        GObject.ParamFlags.READWRITE,
    ),
    'pass-through-mode': (
        bool,
        'Run module in a pass-through mode.',
        'Run module in a pass-through mode. Store frame content in VideoFrame '
        'object as an internal VideoFrameContent.',
        False,
        GObject.ParamFlags.READWRITE,
    ),
    'ingress-module': (
        str,
        'Ingress filter python module.',
        'Name or path of the python module where '
        'the ingress filter class code is located.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'ingress-class': (
        str,
        'Ingress filter python class name.',
        'Name of the python class that implements ingress filter.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'ingress-kwargs': (
        str,
        'Ingress filter init kwargs.',
        'Keyword arguments for ingress filter initialization.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'ingress-dev-mode': (
        bool,
        'Ingress filter dev mode flag.',
        (
            'Whether to monitor the ingress filter source file changes at runtime'
            ' and reload the pyfunc objects when necessary.'
        ),
        False,
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
        self.store_metadata = False
        self.max_parallel_streams: int = 0
        self.video_pipeline: Optional[VideoPipeline] = None
        self.pipeline_stage_name: Optional[str] = None
        self.shutdown_auth: Optional[str] = None
        self.max_width: int = 0
        self.max_height: int = 0
        self.pass_through_mode = False

        self.ingress_module: Optional[str] = None
        self.ingress_class_name: Optional[str] = None
        self.ingress_kwargs: Optional[str] = None
        self.ingress_dev_mode: bool = False
        self.ingress_pyfunc: Optional[PyFunc] = None

        self._frame_idx_gen = itertools.count()

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
            if self.ingress_module and self.ingress_class_name:
                self.ingress_pyfunc = init_pyfunc(
                    self,
                    self.logger,
                    self.ingress_module,
                    self.ingress_class_name,
                    self.ingress_kwargs,
                    self.ingress_dev_mode,
                )
            else:
                # for AO RTSP
                self.logger.debug('Ingress filter is not set, using default one.')
                self.ingress_pyfunc = DefaultIngressFilter()

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
        if prop.name == 'shutdown-auth':
            return self.shutdown_auth
        if prop.name == 'max-width':
            return self.max_width
        if prop.name == 'max-height':
            return self.max_height
        if prop.name == 'pass-through-mode':
            return self.pass_through_mode
        if prop.name == 'ingress-module':
            return self.ingress_module
        if prop.name == 'ingress-class':
            return self.ingress_class_name
        if prop.name == 'ingress-kwargs':
            return self.ingress_kwargs
        if prop.name == 'ingress-dev-mode':
            return self.ingress_dev_mode
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
        elif prop.name == 'shutdown-auth':
            self.shutdown_auth = value
        elif prop.name == 'max-width':
            self.max_width = value
        elif prop.name == 'max-height':
            self.max_height = value
        elif prop.name == 'pass-through-mode':
            self.pass_through_mode = value
        elif prop.name == 'ingress-module':
            self.ingress_module = value
        elif prop.name == 'ingress-class':
            self.ingress_class_name = value
        elif prop.name == 'ingress-kwargs':
            self.ingress_kwargs = value
        elif prop.name == 'ingress-dev-mode':
            self.ingress_dev_mode = value
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
        if not self.is_running:
            self.logger.info(
                'Demuxer is not running. Skipping buffer with timestamp %s.',
                buffer.pts,
            )
            return Gst.FlowReturn.OK

        message = load_message_from_gst_buffer(buffer)
        message.validate_seq_id()
        # TODO: Pipeline message types might be extended beyond only VideoFrame
        # Additional checks for audio/raw_tensors/etc. may be required
        if message.is_video_frame():
            result = self.handle_video_frame(
                message.as_video_frame(),
                message.span_context,
                buffer,
            )
        elif message.is_end_of_stream():
            result = self.handle_eos(message.as_end_of_stream())
        elif message.is_shutdown():
            result = self.handle_shutdown(message.as_shutdown())
        else:
            self.logger.warning('Unsupported message type for message %r', message)
            result = Gst.FlowReturn.OK

        return result

    def handle_video_frame(
        self,
        video_frame: VideoFrame,
        span_context: PropagatedContext,
        buffer: Gst.Buffer,
    ) -> Gst.FlowReturn:
        """Handle VideoFrame message."""

        frame_params = FrameParams.from_video_frame(video_frame)
        frame_pts = convert_ts(video_frame.pts, video_frame.time_base)
        frame_dts = (
            convert_ts(video_frame.dts, video_frame.time_base)
            if video_frame.dts is not None
            else Gst.CLOCK_TIME_NONE
        )
        frame_duration = (
            convert_ts(video_frame.duration, video_frame.time_base)
            if video_frame.duration is not None
            else Gst.CLOCK_TIME_NONE
        )
        self.logger.debug(
            'Received frame %s/%s from source %s; frame %s a keyframe',
            frame_pts,
            frame_dts,
            video_frame.source_id,
            'is' if video_frame.keyframe else 'is not',
        )

        try:
            if not self.ingress_pyfunc(video_frame):
                self.logger.debug(
                    'Frame %s from source %s didnt pass ingress filter, skipping it.',
                    frame_pts,
                    video_frame.source_id,
                )
                return Gst.FlowReturn.OK

            self.logger.debug(
                'Frame %s from source %s passed ingress filter.',
                frame_pts,
                video_frame.source_id,
            )
        except Exception as exc:
            handle_non_fatal_error(
                self,
                self.logger,
                exc,
                f'Error in ingress filter call {self.ingress_pyfunc}',
                self.ingress_dev_mode,
            )
            if video_frame.content.is_none():
                self.logger.debug(
                    'Frame %s from source %s has no content, skipping it.',
                    frame_pts,
                    video_frame.source_id,
                )
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
                    return Gst.FlowReturn.ERROR
                if self.is_greater_than_max_resolution(frame_params):
                    return Gst.FlowReturn.OK
                if not video_frame.keyframe:
                    self.logger.warning(
                        'Frame %s from source %s is not a keyframe, skipping it. '
                        'Stream should start with a keyframe.',
                        frame_pts,
                        video_frame.source_id,
                    )
                    return Gst.FlowReturn.OK
                source_info = SourceInfo(video_frame.source_id, frame_params)
                self.sources[video_frame.source_id] = source_info
            source_info.timestamp = time.time()
        if source_info.src_pad is not None and source_info.params != frame_params:
            if self.is_greater_than_max_resolution(frame_params):
                self.send_eos(source_info)
            self.update_frame_params(source_info, frame_params)
        if source_info.src_pad is not None:
            self.check_timestamps(source_info, frame_pts, frame_dts)
        if frame_pts != Gst.CLOCK_TIME_NONE:
            source_info.last_pts = frame_pts
        if frame_dts != Gst.CLOCK_TIME_NONE:
            source_info.last_dts = frame_dts
        if source_info.src_pad is None:
            if video_frame.keyframe:
                self.add_source(video_frame.source_id, source_info)
            else:
                self.logger.warning(
                    'Frame %s from source %s is not a keyframe, skipping it. '
                    'Stream should start with a keyframe.',
                    frame_pts,
                    video_frame.source_id,
                )
                return Gst.FlowReturn.OK

        try:
            frame_buf = self._build_frame_buffer(video_frame, buffer)
        except ValueError as e:
            self.is_running = False
            error = (
                f'Failed to build buffer for video frame {frame_pts} '
                f'from source {video_frame.source_id}: {e}'
            )
            self.logger.error(error)
            frame = inspect.currentframe()
            gst_post_stream_demux_error(
                gst_element=self,
                frame=frame,
                file_path=__file__,
                text=error,
            )
            return Gst.FlowReturn.ERROR

        frame_idx = self._add_frame_to_pipeline(video_frame, span_context)
        if self.pass_through_mode and not video_frame.content.is_internal():
            content = frame_buf.extract_dup(0, frame_buf.get_size())
            self.logger.debug(
                'Storing content of frame with IDX %s as an internal VideoFrameContent (%s) bytes.',
                frame_idx,
                len(content),
            )
            video_frame.content = VideoFrameContent.internal(content)
        frame_buf.pts = frame_pts
        frame_buf.dts = frame_dts
        frame_buf.duration = (
            Gst.CLOCK_TIME_NONE if frame_duration is None else frame_duration
        )
        self.add_frame_meta(frame_idx, frame_buf, video_frame)
        self.logger.debug('Pushing frame with idx=%s and pts=%s', frame_idx, frame_pts)
        result: Gst.FlowReturn = source_info.src_pad.push(frame_buf)

        self.logger.debug(
            'Frame from source %s with PTS %s was processed.',
            video_frame.source_id,
            frame_pts,
        )
        return result

    def _build_frame_buffer(
        self,
        video_frame: VideoFrame,
        buffer: Gst.Buffer,
    ) -> Gst.Buffer:
        if video_frame.content.is_internal():
            return Gst.Buffer.new_wrapped(video_frame.content.get_data_as_bytes())

        frame_type = ExternalFrameType(video_frame.content.get_method())
        if frame_type != ExternalFrameType.ZEROMQ:
            raise ValueError(f'Unsupported frame type "{frame_type.value}".')
        if buffer.n_memory() < 2:
            raise ValueError(
                f'Buffer has {buffer.n_memory()} regions of memory, expected at least 2.'
            )

        frame_buf: Gst.Buffer = Gst.Buffer.new()
        frame_buf.append_memory(buffer.get_memory_range(1, -1))

        return frame_buf

    def _add_frame_to_pipeline(
        self,
        video_frame: VideoFrame,
        span_context: PropagatedContext,
    ) -> int:
        """Add frame to the pipeline and return frame ID."""

        if self.video_pipeline is not None:
            if span_context.as_dict():
                frame_idx = self.video_pipeline.add_frame_with_telemetry(
                    self.pipeline_stage_name,
                    video_frame,
                    span_context.nested_span(self.video_pipeline.root_span_name),
                )
                self.logger.debug(
                    'Frame with PTS %s from source %s was added to the pipeline '
                    'with telemetry. Frame ID is %s.',
                    video_frame.pts,
                    video_frame.source_id,
                    frame_idx,
                )
            else:
                frame_idx = self.video_pipeline.add_frame(
                    self.pipeline_stage_name,
                    video_frame,
                )
                self.logger.debug(
                    'Frame with PTS %s from source %s was added to the pipeline. '
                    'Frame ID is %s.',
                    video_frame.pts,
                    video_frame.source_id,
                    frame_idx,
                )
        else:
            frame_idx = next(self._frame_idx_gen)
            self.logger.debug(
                'Pipeline is not set, generated ID for frame with PTS %s from '
                'source %s is %s.',
                video_frame.pts,
                video_frame.source_id,
                frame_idx,
            )

        return frame_idx

    def handle_eos(self, eos: EndOfStream) -> Gst.FlowReturn:
        """Handle EndOfStream message."""
        self.logger.info('Received EOS from source %s.', eos.source_id)
        with self.source_lock:
            source_info: SourceInfo = self.sources.get(eos.source_id)
            if source_info is None:
                return Gst.FlowReturn.OK
            source_info.timestamp = time.time()

        if source_info.src_pad is not None:
            self.send_eos(source_info)
        del self.sources[eos.source_id]

        return Gst.FlowReturn.OK

    def handle_shutdown(self, shutdown: Shutdown) -> Gst.FlowReturn:
        """Handle Shutdown message."""
        if self.shutdown_auth is None:
            self.logger.debug('Ignoring shutdown message: shutting down in disabled.')
            return Gst.FlowReturn.OK
        if shutdown.auth != self.shutdown_auth:
            self.logger.debug(
                'Ignoring shutdown message: incorrect authentication key.'
            )
            return Gst.FlowReturn.OK

        self.logger.info('Received shutdown message.')
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

    def check_timestamps(
        self,
        source_info: SourceInfo,
        pts: int,
        dts: int,
    ):
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
            pts,
            source_info.last_dts,
            dts,
        )
        reset = False
        if dts != Gst.CLOCK_TIME_NONE:
            reset = dts < source_info.last_dts
        if pts != Gst.CLOCK_TIME_NONE:
            if dts == Gst.CLOCK_TIME_NONE:
                reset = pts < source_info.last_pts

        if reset:
            self.logger.info(
                'Timestamps on source %s non-monotonous. Resetting source.',
                source_info.source_id,
            )
            self.send_eos(source_info)

    def is_greater_than_max_resolution(self, video_frame: FrameParams) -> bool:
        """Check if the resolution of the incoming stream is greater than the
        max allowed resolution. Return True if the resolution is greater than
        the max allowed resolution, otherwise False.
        """
        if self.max_width and self.max_height:
            if (
                int(video_frame.width) > self.max_width
                or int(video_frame.height) > self.max_height
            ):
                self.logger.warning(
                    f'The resolution of the incoming stream is '
                    f'{video_frame.width}x{video_frame.height} and '
                    f'treater than the allowed max '
                    f'{self.max_width}x'
                    f'{self.max_height}'
                    f' resolutions. Terminate. You can override the max allowed '
                    f"resolution with 'MAX_RESOLUTION' environment variable."
                )
                return True
        return False

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

    def add_frame_meta(self, idx: int, frame_buf: Gst.Buffer, video_frame: VideoFrame):
        """Store metadata of a frame."""
        if self.video_pipeline is not None:
            from pygstsavantframemeta import gst_buffer_add_savant_frame_meta

            if not video_frame.transformations:
                video_frame.add_transformation(
                    VideoFrameTransformation.initial_size(
                        video_frame.width, video_frame.height
                    )
                )
            gst_buffer_add_savant_frame_meta(frame_buf, idx)


# register plugin
GObject.type_register(SavantRsVideoDemux)
__gstelementfactory__ = (
    SavantRsVideoDemux.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SavantRsVideoDemux,
)
