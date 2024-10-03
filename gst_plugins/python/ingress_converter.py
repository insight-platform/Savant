"""GStreamer plugin to execute user-defined ingress converter.

Can be used for modifying frame resolution, content, etc.
"""
from threading import Event, Lock
from typing import Any, Optional

from pygstsavantframemeta import (
    gst_buffer_add_savant_frame_meta,
    gst_buffer_get_savant_frame_meta,
)
from savant_rs.pipeline2 import VideoPipeline

# from savant.deepstream.ingress_converter import BaseIngressConverter
from savant.gstreamer import GLib, GObject, Gst, GstBase  # noqa: F401
from savant.utils.logging import LoggerMixin

# from savant.deepstream.opencv_utils import nvds_to_gpu_mat


# RGBA format is required to access the frame (pyds.get_nvds_buf_surface)
CAPS = Gst.Caps.from_string(
    'video/x-raw(memory:NVMM), '
    'format={RGBA}, '
    f'width={Gst.IntRange(range(1, GLib.MAXINT))}, '
    f'height={Gst.IntRange(range(1, GLib.MAXINT))}, '
    f'framerate={Gst.FractionRange(Gst.Fraction(0, 1), Gst.Fraction(GLib.MAXINT, 1))}'
)

SINK_ORIG_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink_orig',
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    CAPS,
)
SINK_CONV_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink_conv',
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    CAPS,
)
SRC_PAD_TEMPLATE = Gst.PadTemplate.new(
    'src',
    Gst.PadDirection.SRC,
    Gst.PadPresence.ALWAYS,
    CAPS,
)


class IngressConverter(LoggerMixin, Gst.Element):
    """IngressConverter GStreamer plugin."""

    GST_PLUGIN_NAME: str = 'ingress_converter'

    __gstmetadata__ = (
        'GStreamer plugin to execute user-defined ingress converter',
        'Transform',
        'Provides a callback to execute user-defined ingress converter on every frame. '
        'Can be used for modifying frame resolution, content, etc.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        SINK_ORIG_PAD_TEMPLATE,
        SINK_CONV_PAD_TEMPLATE,
        SRC_PAD_TEMPLATE,
    )

    __gproperties__ = {
        'source-id': (
            str,
            'Source ID',
            'Source ID',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'converter': (
            object,
            'Ingress converter instance',
            'Ingress converter instance',
            GObject.ParamFlags.READWRITE,
        ),
        'pipeline': (
            object,
            'VideoPipeline object from savant-rs.',
            'VideoPipeline object from savant-rs.',
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        # properties

        self._source_id: Optional[str] = None
        self._converter: Optional['BaseIngressConverter'] = None
        self._video_pipeline: Optional[VideoPipeline] = None
        # TODO: depends on arch
        # self._memory_type = int(pyds.NVBUF_MEM_DEFAULT)
        self._memory_type = 3

        self._sink_orig_pad: Gst.Pad = Gst.Pad.new_from_template(
            SINK_ORIG_PAD_TEMPLATE,
            'sink_orig',
        )
        self._sink_conv_pad: Gst.Pad = Gst.Pad.new_from_template(
            SINK_CONV_PAD_TEMPLATE,
            'sink_conv',
        )
        self._srcpad: Gst.Pad = Gst.Pad.new_from_template(
            SRC_PAD_TEMPLATE,
            'src',
        )

        self.add_pad(self._sink_orig_pad)
        self.add_pad(self._sink_conv_pad)
        self.add_pad(self._srcpad)

        self._sink_orig_pad.set_chain_function_full(self.handle_buffer)
        self._sink_conv_pad.set_chain_function_full(self.handle_buffer)

        self._pending_orig_buffer: Optional[Gst.Buffer] = None
        self._pending_orig_buffer_lock = Event()
        self._pending_orig_buffer_lock.set()
        self._pending_conv_buffer: Optional[Gst.Buffer] = None
        self._pending_conv_buffer_lock = Event()
        self._pending_conv_buffer_lock.set()

        self._lock = Lock()

        self._sink_conv_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            self.on_sink_pad_event,
        )
        self._sink_orig_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            self.on_sink_pad_event,
        )

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        self.logger.info('Getting property %s', prop.name)
        if prop.name == 'source-id':
            return self._source_id
        if prop.name == 'converter':
            return self._converter
        if prop.name == 'pipeline':
            return self._video_pipeline
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        self.logger.info('Setting property %s to %s', prop.name, value)
        if prop.name == 'source-id':
            self._source_id = value
            return
        if prop.name == 'converter':
            self._converter = value
            return
        elif prop.name == 'pipeline':
            self._video_pipeline = value
            return
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def handle_buffer(
        self,
        sink_pad: Gst.Pad,
        element: Gst.Element,
        buffer: Gst.Buffer,
    ) -> Gst.FlowReturn:
        self.logger.info(
            '[%s] Handling buffer %s (%s)',
            sink_pad.get_name(),
            buffer.pts,
            buffer,
        )

        if sink_pad == self._sink_orig_pad:
            # TODO: configure
            while not self._pending_orig_buffer_lock.wait(1):
                pass
            with self._lock:
                self._pending_orig_buffer_lock.clear()
                self._pending_orig_buffer = buffer

        else:
            # TODO: configure
            while not self._pending_conv_buffer_lock.wait(1):
                pass
            with self._lock:
                self._pending_conv_buffer_lock.clear()
                self._pending_conv_buffer = buffer

        self.logger.info(
            '[%s] Buffer %s (%s) is pending',
            sink_pad.get_name(),
            buffer.pts,
            buffer,
        )

        with self._lock:
            if (
                self._pending_orig_buffer is not None
                and self._pending_conv_buffer is not None
            ):
                res = self.convert(self._pending_orig_buffer, self._pending_conv_buffer)
                self._pending_orig_buffer = None
                self._pending_orig_buffer_lock.set()
                self._pending_conv_buffer = None
                self._pending_conv_buffer_lock.set()
                return res
            else:
                self.logger.info(
                    '[%s] Waiting for another buffer',
                    sink_pad.get_name(),
                )

        return Gst.FlowReturn.OK

    def convert(
        self,
        orig_buffer: Gst.Buffer,
        conv_buffer: Gst.Buffer,
    ) -> Gst.FlowReturn:
        from savant.deepstream.opencv_utils import nvds_to_gpu_mat

        self.logger.info(
            'Converting buffer %s (%s) to %s (%s)',
            orig_buffer.pts,
            orig_buffer,
            conv_buffer.pts,
            conv_buffer,
        )

        savant_frame_meta = gst_buffer_get_savant_frame_meta(orig_buffer)
        if savant_frame_meta is None:
            self.logger.error(
                'No Savant Frame Metadata found on buffer with PTS %s.',
                orig_buffer.pts,
            )
            return Gst.FlowReturn.ERROR

        video_frame, _ = self._video_pipeline.get_independent_frame(
            savant_frame_meta.idx
        )

        with nvds_to_gpu_mat(orig_buffer, batch_id=0) as in_mat:
            with nvds_to_gpu_mat(conv_buffer, batch_id=0) as out_mat:
                # TODO: use opencv stream?
                self._converter.convert(self._source_id, video_frame, in_mat, out_mat)

        conv_buffer.pts = orig_buffer.pts
        conv_buffer.dts = orig_buffer.dts
        conv_buffer.duration = orig_buffer.duration
        gst_buffer_add_savant_frame_meta(conv_buffer, savant_frame_meta.idx)

        return self._srcpad.push(conv_buffer)

    def on_sink_pad_event(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ) -> Gst.PadProbeReturn:
        event: Gst.Event = info.get_event()
        self.logger.info(
            'Sink pad %s received event %s',
            pad.get_name(),
            event.type,
        )
        if pad == self._sink_orig_pad and event.type == Gst.EventType.CAPS:
            self.logger.info(
                'Sink pad %s dropping event %s',
                pad.get_name(),
                event.type,
            )
            return Gst.PadProbeReturn.DROP

        self._srcpad.push_event(event)

        return Gst.PadProbeReturn.OK


# register plugin
GObject.type_register(IngressConverter)
__gstelementfactory__ = (
    IngressConverter.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    IngressConverter,
)
