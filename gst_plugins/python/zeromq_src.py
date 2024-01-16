"""ZeroMQ src."""
import inspect
from typing import Optional, Tuple

from pygstsavantframemeta import gst_buffer_add_savant_frame_meta
from savant_rs.pipeline2 import VideoPipeline
from savant_rs.primitives import (
    EndOfStream,
    Shutdown,
    VideoFrame,
    VideoFrameContent,
    VideoFrameTransformation,
)
from savant_rs.utils import PropagatedContext
from savant_rs.utils.serialization import Message, load_message_from_bytes

from gst_plugins.python.pyfunc_common import handle_non_fatal_error, init_pyfunc
from gst_plugins.python.zeromq_properties import ZEROMQ_PROPERTIES, socket_type_property
from savant.api.enums import ExternalFrameType
from savant.api.parser import convert_ts
from savant.base.frame_filter import DefaultIngressFilter
from savant.base.pyfunc import PyFunc
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.event import build_savant_eos_event
from savant.gstreamer.utils import (
    gst_post_library_settings_error,
    gst_post_stream_failed_error,
    required_property,
)
from savant.utils.logging import LoggerMixin
from savant.utils.zeromq import (
    Defaults,
    ReceiverSocketTypes,
    ZeroMQSource,
    ZMQException,
    build_topic_prefix,
)

HandlerResult = Optional[Tuple[Gst.FlowReturn, Optional[Gst.Buffer]]]

ZEROMQ_SRC_PROPERTIES = {
    **ZEROMQ_PROPERTIES,
    'socket-type': socket_type_property(ReceiverSocketTypes),
    'receive-hwm': (
        int,
        'High watermark for inbound messages',
        'High watermark for inbound messages',
        1,
        GObject.G_MAXINT,
        Defaults.RECEIVE_HWM,
        GObject.ParamFlags.READWRITE,
    ),
    'source-id': (
        str,
        'Source ID filter.',
        'Filter frames by source ID.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'source-id-prefix': (
        str,
        'Source ID prefix filter.',
        'Filter frames by source ID prefix.',
        None,
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


class ZeromqSrc(LoggerMixin, GstBase.BaseSrc):
    """ZeromqSrc GstPlugin."""

    GST_PLUGIN_NAME = 'zeromq_src'

    __gstmetadata__ = (
        'ZeroMQ source',
        'Source',
        'Reads binary messages from zeromq',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = Gst.PadTemplate.new(
        'src',
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.new_any(),
    )

    __gproperties__ = ZEROMQ_SRC_PROPERTIES

    def __init__(self):
        GstBase.BaseSrc.__init__(self)

        # properties
        self.socket: str = None
        self.socket_type: str = ReceiverSocketTypes.ROUTER.name
        self.bind: bool = True
        self.receive_timeout: int = Defaults.RECEIVE_TIMEOUT
        self.receive_hwm: int = Defaults.RECEIVE_HWM
        self.source_id: Optional[str] = None
        self.source_id_prefix: Optional[str] = None
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

        self.source: ZeroMQSource = None
        self.set_live(True)

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: property parameters
        """

        if prop.name == 'socket':
            return self.socket
        if prop.name == 'socket-type':
            return self.socket_type
        if prop.name == 'bind':
            return self.bind
        if prop.name == 'receive-hwm':
            return self.receive_hwm
        if prop.name == 'source-id':
            return self.source_id
        if prop.name == 'source-id-prefix':
            return self.source_id_prefix

        if prop.name == 'pipeline':
            return self.video_pipeline
        if prop.name == 'pipeline-stage-name':
            return self.pipeline_stage_name
        if prop.name == 'shutdown-auth':
            return self.shutdown_auth
        if prop.name == 'pass-through-mode':
            return self.pass_through_mode

        if prop.name == 'max-width':
            return self.max_width
        if prop.name == 'max-height':
            return self.max_height

        if prop.name == 'ingress-module':
            return self.ingress_module
        if prop.name == 'ingress-class':
            return self.ingress_class_name
        if prop.name == 'ingress-kwargs':
            return self.ingress_kwargs
        if prop.name == 'ingress-dev-mode':
            return self.ingress_dev_mode

        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: property parameters
        :param value: new value for param, type dependents on param
        """
        self.logger.debug('Setting property "%s" to "%s".', prop.name, value)
        if prop.name == 'socket':
            self.socket = value
        elif prop.name == 'socket-type':
            self.socket_type = value
        elif prop.name == 'bind':
            self.bind = value
        elif prop.name == 'receive-hwm':
            self.receive_hwm = value
        elif prop.name == 'source-id':
            self.source_id = value
        elif prop.name == 'source-id-prefix':
            self.source_id_prefix = value

        elif prop.name == 'pipeline':
            self.video_pipeline = value
        elif prop.name == 'pipeline-stage-name':
            self.pipeline_stage_name = value
        elif prop.name == 'shutdown-auth':
            self.shutdown_auth = value
        elif prop.name == 'pass-through-mode':
            self.pass_through_mode = value

        elif prop.name == 'max-width':
            self.max_width = value
        elif prop.name == 'max-height':
            self.max_height = value

        elif prop.name == 'ingress-module':
            self.ingress_module = value
        elif prop.name == 'ingress-class':
            self.ingress_class_name = value
        elif prop.name == 'ingress-kwargs':
            self.ingress_kwargs = value
        elif prop.name == 'ingress-dev-mode':
            self.ingress_dev_mode = value

        else:
            raise AttributeError(f'Unknown property "{prop.name}".')

    def do_start(self):
        """Start source."""
        self.logger.debug('Starting ZeroMQ source')

        try:
            required_property('socket', self.socket)
            required_property('pipeline', self.video_pipeline)
            required_property('pipeline-stage-name', self.pipeline_stage_name)
            topic_prefix = build_topic_prefix(self.source_id, self.source_id_prefix)

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

            self.source = ZeroMQSource(
                socket=self.socket,
                socket_type=self.socket_type,
                bind=self.bind,
                receive_timeout=self.receive_timeout,
                receive_hwm=self.receive_hwm,
                topic_prefix=topic_prefix,
            )
        except Exception as exc:
            error = f'Failed to start ZeroMQ source with socket {self.socket}: {exc}.'
            self.logger.exception(error, exc_info=True)
            frame = inspect.currentframe()
            gst_post_library_settings_error(self, frame, __file__, error)
            # prevents pipeline from starting
            return False

        return True

    def start_zero_mq_source(self):
        try:
            self.source.start()
        except ZMQException:
            error = f'Failed to start ZeroMQ source with socket {self.socket}.'
            self.logger.exception(error, exc_info=True)
            frame = inspect.currentframe()
            gst_post_stream_failed_error(
                gst_element=self,
                frame=frame,
                file_path=__file__,
                text=error,
            )
            gst_post_library_settings_error(self, frame, __file__)
            return False
        return True

    # pylint: disable=unused-argument
    def do_create(self, offset: int, size: int, buffer: Gst.Buffer = None):
        """Create gst buffer."""

        self.logger.debug('Creating next buffer')

        if not self.source.is_alive:
            if not self.start_zero_mq_source():
                return Gst.FlowReturn.ERROR

        result = None
        while result is None:
            flow_return = self.wait_playing()
            if flow_return != Gst.FlowReturn.OK:
                self.logger.info('Returning %s', flow_return)
                return flow_return, None
            result = self.try_create()

        return result

    def try_create(self) -> HandlerResult:
        zmq_message = self.source.next_message()
        if zmq_message is None:
            return
        self.logger.debug('Received message of sizes %s', [len(x) for x in zmq_message])
        message = load_message_from_bytes(zmq_message[0])
        if len(zmq_message) < 2:
            external_content = None
        elif len(zmq_message) == 2:
            external_content = zmq_message[1]
        else:
            external_content = b''.join(zmq_message[1:])

        return self.handle_message(message, external_content)

    def handle_message(
        self,
        message: Message,
        external_content: Optional[bytes],
    ) -> HandlerResult:
        message.validate_seq_id()
        if message.is_video_frame():
            return self.handle_video_frame(
                message.as_video_frame(),
                message.span_context,
                external_content,
            )
        if message.is_end_of_stream():
            return self.handle_eos(message.as_end_of_stream())
        if message.is_shutdown():
            return self.handle_shutdown(message.as_shutdown())
        self.logger.warning('Unsupported message type for message %r', message)

    def handle_video_frame(
        self,
        video_frame: VideoFrame,
        span_context: PropagatedContext,
        external_content: Optional[bytes],
    ) -> HandlerResult:
        """Handle VideoFrame message."""

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

        if self.is_greater_than_max_resolution(video_frame):
            return

        try:
            if not self.ingress_pyfunc(video_frame):
                self.logger.debug(
                    'Frame %s from source %s didnt pass ingress filter, skipping it.',
                    frame_pts,
                    video_frame.source_id,
                )
                return

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
                return

        try:
            frame_buf = self.build_frame_buffer(video_frame, external_content)
        except ValueError as e:
            error = (
                f'Failed to build buffer for video frame {frame_pts} '
                f'from source {video_frame.source_id}: {e}'
            )
            self.logger.error(error)
            frame = inspect.currentframe()
            gst_post_stream_failed_error(
                gst_element=self,
                frame=frame,
                file_path=__file__,
                text=error,
            )
            return Gst.FlowReturn.ERROR, None

        frame_idx = self.add_frame_to_pipeline(video_frame, span_context)
        if self.pass_through_mode and not video_frame.content.is_internal():
            self.logger.debug(
                'Storing content of frame with IDX %s as an internal VideoFrameContent (%s) bytes.',
                frame_idx,
                len(external_content),
            )
            video_frame.content = VideoFrameContent.internal(external_content)
        frame_buf.pts = frame_pts
        frame_buf.dts = frame_dts
        frame_buf.duration = frame_duration
        self.add_frame_meta(frame_idx, frame_buf, video_frame)
        self.logger.debug(
            'Frame with PTS %s from source %s has been processed.',
            frame_pts,
            video_frame.source_id,
        )

        return Gst.FlowReturn.OK, frame_buf

    def build_frame_buffer(
        self,
        video_frame: VideoFrame,
        external_content: Optional[bytes],
    ) -> Gst.Buffer:
        if video_frame.content.is_internal():
            return Gst.Buffer.new_wrapped(video_frame.content.get_data_as_bytes())

        frame_type = ExternalFrameType(video_frame.content.get_method())
        if frame_type != ExternalFrameType.ZEROMQ:
            raise ValueError(f'Unsupported frame type "{frame_type.value}".')

        if not external_content:
            raise ValueError(
                f'Frame with PTS {video_frame.pts} from source '
                f'{video_frame.source_id} has no external content.'
            )

        return Gst.Buffer.new_wrapped(external_content)

    def add_frame_to_pipeline(
        self,
        video_frame: VideoFrame,
        span_context: PropagatedContext,
    ) -> int:
        """Add frame to the pipeline and return frame ID."""

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

        return frame_idx

    def add_frame_meta(self, idx: int, frame_buf: Gst.Buffer, video_frame: VideoFrame):
        """Store metadata of a frame."""

        if not video_frame.transformations:
            video_frame.add_transformation(
                VideoFrameTransformation.initial_size(
                    video_frame.width, video_frame.height
                )
            )
        gst_buffer_add_savant_frame_meta(frame_buf, idx)

    def handle_eos(self, eos: EndOfStream) -> HandlerResult:
        """Handle EndOfStream message."""

        self.logger.info('Received EOS from source %s.', eos.source_id)
        savant_eos_event = build_savant_eos_event(eos.source_id)
        result = self.srcpad.push_event(savant_eos_event)
        if result != Gst.FlowReturn.OK:
            self.logger.error(
                'Failed to push savant-eos event to the pipeline (%s)',
                result,
            )
            return result, None

    def handle_shutdown(self, shutdown: Shutdown) -> HandlerResult:
        """Handle Shutdown message."""

        if self.shutdown_auth is None:
            self.logger.debug('Ignoring shutdown message: shutting down in disabled.')
            return
        if shutdown.auth != self.shutdown_auth:
            self.logger.debug(
                'Ignoring shutdown message: incorrect authentication key.'
            )
            return

        self.logger.info('Received shutdown message: sending EOS.')
        self.srcpad.push_event(Gst.Event.new_eos())
        return Gst.FlowReturn.EOS, None

    def do_stop(self):
        """Gst src stop callback."""
        self.source.terminate()
        return True

    def do_is_seekable(self):
        """Check if the source can seek."""
        return False

    def is_greater_than_max_resolution(self, video_frame: VideoFrame) -> bool:
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
                    f'greater than the allowed max '
                    f'{self.max_width}x'
                    f'{self.max_height}'
                    f' resolutions. Terminate. You can override the max allowed '
                    f"resolution with 'MAX_RESOLUTION' environment variable."
                )
                return True
        return False


# register plugin
GObject.type_register(ZeromqSrc)
__gstelementfactory__ = (
    ZeromqSrc.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    ZeromqSrc,
)
