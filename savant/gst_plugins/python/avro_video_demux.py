"""AvroVideoDemux element."""
import itertools

import time
from fractions import Fraction
from threading import Lock, Thread
from typing import Dict, NamedTuple, Optional

from dataclasses import dataclass

from savant.api import deserialize
from savant.gstreamer import GObject, Gst
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec
from savant.gstreamer.metadata import (
    DEFAULT_FRAMERATE,
    SourceFrameMeta,
    metadata_add_frame_meta,
    OnlyExtendedDict,
)
from savant.gstreamer.utils import LoggerMixin

DEFAULT_SOURCE_TIMEOUT = 60
DEFAULT_SOURCE_EVICTION_INTERVAL = 15
OUT_CAPS = Gst.Caps.from_string(';'.join(x.value.caps_with_params for x in Codec))

AVRO_VIDEO_DEMUX_PROPERTIES = {
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
    'store-metadata': (
        bool,
        'Store metadata',
        'Store metadata',
        False,
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
    # TODO: filter frames by source id in zeromq_src
    #       https://github.com/insight-platform/Savant/issues/59
    'source-id': (
        str,
        'Source ID filter.',
        'Filter frames by source ID.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
}


class FrameParams(NamedTuple):
    """Frame parameters."""

    codec: Codec
    width: str
    height: str
    framerate: str


@dataclass
class SourceInfo:
    """Info about source/camera."""

    source_id: str
    params: FrameParams
    src_pad: Optional[Gst.Pad] = None
    timestamp: float = 0
    last_pts: int = 0
    last_dts: int = 0


class AvroVideoDemux(LoggerMixin, Gst.Element):
    """AvroVideoDemux GstPlugin."""

    GST_PLUGIN_NAME = 'avro_video_demux'

    __gstmetadata__ = (
        'Avro video demuxer',
        'Demuxer',
        'Deserializes avro video frames and demultiplex them by source ID. '
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

    __gproperties__ = AVRO_VIDEO_DEMUX_PROPERTIES

    def __init__(self):
        super().__init__()
        self.sources: Dict[str, SourceInfo] = {}
        self.eos_on_timestamps_reset = False
        self.source_timeout = DEFAULT_SOURCE_TIMEOUT
        self.source_eviction_interval = DEFAULT_SOURCE_EVICTION_INTERVAL
        self.last_eviction = 0
        self.source_lock = Lock()
        self.expiration_thread = Thread(target=self.eviction_job, daemon=True)
        self.store_metadata = False
        self.max_parallel_streams: int = 0
        self.source_id: Optional[str] = None

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
        """Start an expiration thread if state changed from NULL."""
        if (
            old == Gst.State.NULL
            and new != Gst.State.NULL
            and not self.expiration_thread.is_alive()
        ):
            self.expiration_thread.start()

    def do_get_property(self, prop):
        """Get property callback."""
        if prop.name == 'source-timeout':
            return self.source_timeout
        if prop.name == 'source-eviction-interval':
            return self.source_eviction_interval
        if prop.name == 'store-metadata':
            return self.store_metadata
        if prop.name == 'eos-on-timestamps-reset':
            return self.eos_on_timestamps_reset
        if prop.name == 'max-parallel-streams':
            return self.max_parallel_streams
        if prop.name == 'source-id':
            return self.source_id
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        """Set property callback."""
        if prop.name == 'source-timeout':
            self.source_timeout = value
        elif prop.name == 'source-eviction-interval':
            self.source_eviction_interval = value
        elif prop.name == 'store-metadata':
            self.store_metadata = value
        elif prop.name == 'eos-on-timestamps-reset':
            self.eos_on_timestamps_reset = value
        elif prop.name == 'max-parallel-streams':
            self.max_parallel_streams = value
        elif prop.name == 'source-id':
            self.source_id = value
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
        message_bin = buffer.extract_dup(0, buffer.get_size())

        # TODO: Pipeline message types might be extended beyond only VideoFrame
        # Additional checks for audio/raw_tensors/etc. may be required
        schema_name, message = deserialize(message_bin)
        if self.source_id is not None and message['source_id'] != self.source_id:
            self.logger.debug('Skipping message from source %s', message['source_id'])
            return Gst.FlowReturn.OK
        if schema_name == 'VideoFrame':
            return self.handle_video_frame(message)
        if schema_name == 'EndOfStream':
            return self.handle_eos(message)
        self.logger.error('Unknown schema "%s"', schema_name)
        return Gst.FlowReturn.ERROR

    def handle_video_frame(self, message: Dict) -> Gst.FlowReturn:
        """Handle VideoFrame message."""
        source_id = message['source_id']
        frame_params = FrameParams(
            codec=CODEC_BY_NAME[message['codec']],
            width=message['width'],
            height=message['height'],
            framerate=message['framerate'],
        )
        frame_pts = message['pts']
        frame_dts = message['dts']
        if frame_dts is None:
            frame_dts = Gst.CLOCK_TIME_NONE
        frame_duration = message['duration']
        frame = message['frame']
        self.logger.debug(
            'Received frame %s from source %s, size: %s bytes',
            frame_pts,
            source_id,
            len(frame) if frame else 0,
        )
        frame_idx = next(self._frame_idx_gen)

        with self.source_lock:
            source_info: SourceInfo = self.sources.get(source_id)
            if source_info is None:
                if (
                    self.max_parallel_streams
                    and len(self.sources) >= self.max_parallel_streams
                ):
                    self.logger.warning(
                        'Reached maximum number of streams: %s. '
                        'Skipping frame %s from source %s',
                        self.max_parallel_streams,
                        frame_pts,
                        source_id,
                    )
                    return Gst.FlowReturn.OK
                source_info = SourceInfo(source_id, frame_params)
                self.sources[source_id] = source_info
            source_info.timestamp = time.time()
        if source_info.src_pad is not None and source_info.params != frame_params:
            self.update_frame_params(source_info, frame_params)
        if source_info.src_pad is not None:
            self.check_timestamps(source_info, frame_pts, frame_dts)
        source_info.last_pts = frame_pts
        source_info.last_dts = frame_dts
        if source_info.src_pad is None:
            self.add_source(source_id, source_info)

        if frame:
            frame_buf = Gst.Buffer.new_wrapped(frame)
            frame_buf.pts = frame_pts
            frame_buf.dts = frame_dts
            frame_buf.duration = (
                Gst.CLOCK_TIME_NONE if frame_duration is None else frame_duration
            )
            self.add_frame_meta(frame_idx, frame_buf, message)
            self.logger.debug(
                'Pushing frame with idx=%s and pts=%s', frame_idx, frame_pts
            )
            result: Gst.FlowReturn = source_info.src_pad.push(frame_buf)
        else:
            result = Gst.FlowReturn.OK
        self.logger.debug(
            'end handle_buffer (return buffer with timestamp %d).', frame_pts
        )
        return result

    def handle_eos(self, message: Dict) -> Gst.FlowReturn:
        """Handle EndOfStream message."""
        source_id = message['source_id']
        self.logger.info('Received EOS from source %s.', source_id)
        with self.source_lock:
            source_info: SourceInfo = self.sources.get(source_id)
            if source_info is None:
                return Gst.FlowReturn.OK
            source_info.timestamp = time.time()
        if source_info.src_pad is not None:
            self.send_eos(source_info)
        del self.sources[source_id]
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

    def send_eos(self, source_info: SourceInfo):
        """Send EOS event to a src pad."""
        self.logger.debug(
            'Sending EOS event to pad %s.',
            source_info.src_pad.get_name(),
        )
        source_info.src_pad.push_event(Gst.Event.new_eos())
        self.logger.debug('Removing pad %s', source_info.src_pad.get_name())
        self.remove_pad(source_info.src_pad)
        source_info.src_pad = None

    def eviction_job(self):
        """Eviction job."""
        while True:
            self.eviction_loop()

    def eviction_loop(self):
        """Evicion job loop."""
        self.logger.debug('Start eviction loop')
        with self.source_lock:
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

    def add_frame_meta(self, idx: int, frame_buf: Gst.Buffer, message: Dict):
        """Store metadata of a frame."""
        if self.store_metadata:
            from pygstsavantframemeta import gst_buffer_add_savant_frame_meta

            source_id = message['source_id']
            pts = message['pts']
            frame_meta = SourceFrameMeta(
                source_id=source_id,
                pts=pts,
                duration=message['duration'],
                framerate=message['framerate'],
                metadata=message['metadata'],
                tags=OnlyExtendedDict(message['tags']),
            )
            metadata_add_frame_meta(source_id, idx, pts, frame_meta)
            gst_buffer_add_savant_frame_meta(frame_buf, idx)


def build_caps(params: FrameParams) -> Gst.Caps:
    """Caps factory."""
    try:
        framerate = Fraction(params.framerate)
    except (ZeroDivisionError, ValueError):
        framerate = Fraction(DEFAULT_FRAMERATE)
    framerate = Gst.Fraction(framerate.numerator, framerate.denominator)
    caps = Gst.Caps.from_string(params.codec.value.caps_with_params)
    caps.set_value('width', params.width)
    caps.set_value('height', params.height)
    caps.set_value('framerate', framerate)

    return caps


# register plugin
GObject.type_register(AvroVideoDemux)
__gstelementfactory__ = (
    AvroVideoDemux.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    AvroVideoDemux,
)
