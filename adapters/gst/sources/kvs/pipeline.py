from queue import Empty, Queue
from typing import Optional

from adapters.gst.sinks.video_files import GstPipelineRunner
from adapters.shared.thread import BaseThreadWorker
from savant.gstreamer import Gst, GstApp
from savant.gstreamer.codecs import CODEC_BY_CAPS_NAME, Codec
from savant.gstreamer.utils import on_pad_event

from . import LOGGER_PREFIX
from .config import Config
from .poller import Fragment
from .state import State
from .stream_model import StreamModel

QUEUE_POLL_INTERVAL = 1


class Pipeline(BaseThreadWorker):
    """GStreamer pipeline for sending frames from KVS fragments to a ZMQ socket."""

    def __init__(
        self,
        config: Config,
        stream: StreamModel,
        queue: Queue[Fragment],
        state: Optional[State] = None,
        queue_poll_interval: float = QUEUE_POLL_INTERVAL,
    ):
        super().__init__(
            f'Pipeline-{stream.name}',
            logger_name=f'{LOGGER_PREFIX}.pipeline',
        )
        self.config = config
        self.stream = stream
        self.queue = queue
        self.state = state
        self.queue_poll_interval = queue_poll_interval
        self.pipeline: Optional[Gst.Pipeline] = None
        self.appsrc: Optional[GstApp.AppSrc] = None
        self.runner: Optional[GstPipelineRunner] = None
        self.first_ts: Optional[float] = None
        self.last_ts: Optional[float] = None

    def workload(self):
        """Create and run the pipeline."""

        self.logger.info('Starting pipeline')
        self.pipeline = Gst.parse_launch(
            ' ! '.join(
                [
                    'appsrc name=appsrc emit-signals=false max-buffers=1 block=true',
                    'matroskademux name=demuxer',
                ]
            )
        )
        demuxer = self.pipeline.get_by_name('demuxer')
        demuxer.connect('pad-added', self.on_pad_added)
        self.appsrc = self.pipeline.get_by_name('appsrc')
        self.runner = GstPipelineRunner(self.pipeline)
        self.runner.startup()
        while self.is_running and self.runner.is_running:
            try:
                self.process_fragment()
            except Exception as e:
                self.logger.error(f'Error processing fragment: %s', e, exc_info=True)
                break

        self.is_running = False
        if self.runner.is_running:
            self.logger.info('Sending EOS')
            self.appsrc.end_of_stream()
        self.runner.shutdown()
        self.logger.info('Pipeline stopped')

    def process_fragment(self):
        """Process a fragment from the queue."""

        try:
            fragment = self.queue.get(timeout=self.queue_poll_interval)
        except Empty:
            self.logger.debug('No fragments to process')
            return

        self.logger.debug(
            'Processing fragment %r with %d bytes',
            fragment.fragment_number,
            len(fragment.content),
        )
        self.appsrc.emit('push-buffer', Gst.Buffer.new_wrapped(fragment.content))
        self.queue.task_done()
        if self.first_ts is None:
            self.first_ts = fragment.timestamp
        self.last_ts = fragment.timestamp
        if self.state is not None:
            self.state.update(last_ts=fragment.timestamp)
        self.logger.debug('Processed fragment %r', fragment.fragment_number)

    def on_pad_added(self, element: Gst.Element, pad: Gst.Pad):
        """Handle the pad-added signal on the demuxer."""

        self.logger.info('Received pad-added signal on %s', pad.get_name())
        caps: Gst.Caps = pad.get_current_caps()
        self.logger.info('Caps: %s', caps)
        if caps is None:
            pad.add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                on_pad_event,
                {Gst.EventType.CAPS: self.on_caps},
            )
        else:
            self.add_sink(pad, caps)

    def on_caps(self, pad: Gst.Pad, event: Gst.Event):
        """Handle the caps event on the demuxer."""

        self.logger.info('%s: received event %s', pad.get_name(), event)
        caps = event.parse_caps()
        self.logger.info('Caps: %s', caps)

        try:
            self.add_sink(pad, caps)

        except AssertionError:
            return Gst.PadProbeReturn.ERROR

        return Gst.PadProbeReturn.OK

    def add_sink(self, pad: Gst.Pad, caps: Gst.Caps):
        """Add sink elements to the pipeline."""

        self.logger.debug(f'Try to find codec for %r', caps[0].get_name())
        try:
            codec = CODEC_BY_CAPS_NAME[caps[0].get_name()]
        except KeyError:
            self.logger.error(
                'Pad %s.%s has caps %s. Not attaching parser to it.',
                pad.get_parent().get_name(),
                pad.get_name(),
                caps.to_string(),
            )
            return

        self.logger.info('Adding sink')

        gst_elements = []

        parser: Gst.Element = Gst.ElementFactory.make(codec.value.parser)
        if codec in [Codec.H264, Codec.HEVC]:
            parser.set_property('config-interval', -1)
        gst_elements.append(parser)

        if self.config.fps_meter.period_seconds or self.config.fps_meter.period_frames:
            fps_meter: Gst.Element = Gst.ElementFactory.make('fps_meter')
            fps_meter.set_property('output', self.config.fps_meter.output)
            if self.config.fps_meter.period_seconds:
                fps_meter.set_property(
                    'period-seconds', self.config.fps_meter.period_seconds
                )
            else:
                fps_meter.set_property(
                    'period-frames', self.config.fps_meter.period_frames
                )
            gst_elements.append(fps_meter)

        capsfilter: Gst.Element = Gst.ElementFactory.make('capsfilter')
        capsfilter.set_property(
            'caps',
            Gst.Caps.from_string(codec.value.caps_with_params),
        )
        gst_elements.append(capsfilter)

        if self.config.sync_output:
            adjust_timestamps: Gst.Element = Gst.ElementFactory.make(
                'adjust_timestamps'
            )
            gst_elements.append(adjust_timestamps)

        sink: Gst.Element = Gst.ElementFactory.make('zeromq_sink')
        sink.set_property('socket', self.config.zmq_endpoint)
        sink.set_property('source-id', self.stream.source_id)
        sink.set_property('sync', self.config.sync_output)
        if self.config.sync_output:
            ts_offset = (
                self.pipeline.get_clock().get_time()
                - self.pipeline.get_base_time()
                - int(self.first_ts * Gst.SECOND)
            )
            sink.set_property('ts-offset', ts_offset)
        gst_elements.append(sink)

        last_element = None
        for element in gst_elements:
            self.logger.debug('Adding element %r', element.get_name())
            self.pipeline.add(element)
            if last_element is not None:
                self.logger.debug(
                    'Linking element %r to %r',
                    last_element.get_name(),
                    element.get_name(),
                )
                assert last_element.link(
                    element
                ), f'Failed to link {last_element.get_name()} to {element.get_name()}'
            last_element = element

        parser_pad: Gst.Pad = parser.get_static_pad('sink')
        self.logger.debug('Linking %r to %r', pad.get_name(), parser_pad.get_name())

        for element in gst_elements:
            element.sync_state_with_parent()

        parser_pad.send_event(self.build_stream_name_event())
        assert (
            pad.link(parser_pad) == Gst.PadLinkReturn.OK
        ), f'Failed to link {pad.get_name()} to {parser_pad.get_name()}'

    def build_stream_name_event(self) -> Gst.Event:
        tag_list: Gst.TagList = Gst.TagList.new_empty()
        tag_list.add_value(Gst.TagMergeMode.APPEND, Gst.TAG_LOCATION, self.stream.name)
        return Gst.Event.new_tag(tag_list)
