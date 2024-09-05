import inspect
from threading import Event
from typing import Optional

from savant.gstreamer import GObject, Gst  # noqa:F401
from savant.gstreamer.utils import (
    RequiredPropertyError,
    gst_post_library_settings_error,
    required_property,
)
from savant.utils.logging import LoggerMixin

DEFAULT_MAX_QUEUE = 1
WAIT_QUEUE_INTERVAL = 0.1


class SyncIoWrapperBin(LoggerMixin, Gst.Bin):
    GST_PLUGIN_NAME = 'sync_io_wrapper_bin'

    __gstmetadata__ = (
        'Wrapper for synchronized processing by nested element',
        'Bin',
        'Wrapper for synchronized processing by nested element',
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
            'src',
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        ),
    )

    __gproperties__ = {
        'max-queue-size': (
            GObject.TYPE_UINT,
            'Max queue size',
            'Max queue size',
            1,
            GObject.G_MAXUINT,
            DEFAULT_MAX_QUEUE,
            GObject.ParamFlags.READWRITE,
        ),
        'nested-element': (
            Gst.Element,
            'Nested element',
            'Nested element',
            GObject.ParamFlags.READWRITE | Gst.PARAM_MUTABLE_READY,
        ),
    }

    def __init__(self):
        super().__init__()

        self._sink_pad: Gst.GhostPad = Gst.GhostPad.new_no_target(
            'sink', Gst.PadDirection.SINK
        )
        self.add_pad(self._sink_pad)
        self._sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_sink_buffer)

        self._src_pad: Gst.GhostPad = Gst.GhostPad.new_no_target(
            'src', Gst.PadDirection.SRC
        )
        self._src_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_src_buffer)
        self.add_pad(self._src_pad)

        self._nested_element: Optional[Gst.Element] = None

        self._buffers_in = 0
        self._buffers_out = 0
        self._max_queue_size = DEFAULT_MAX_QUEUE
        self._queue_is_ready = Event()

    def do_state_changed(self, old: Gst.State, new: Gst.State, pending: Gst.State):
        if old == Gst.State.NULL and new != Gst.State.NULL:
            try:
                required_property('nested-element', self._nested_element)
            except RequiredPropertyError as exc:
                self.logger.exception('Failed to start element: %s', exc, exc_info=True)
                frame = inspect.currentframe()
                gst_post_library_settings_error(self, frame, __file__, text=exc.args[0])
                return

    def do_get_property(self, prop):
        if prop.name == 'max-queue-size':
            return self._max_queue_size
        if prop.name == 'nested-element':
            return self._nested_element
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        if prop.name == 'max-queue-size':
            self._max_queue_size = value
        elif prop.name == 'nested-element':
            if self._nested_element is not None:
                self._sink_pad.set_target(None)
                self._src_pad.set_target(None)
                self.remove(self._nested_element)

            self._nested_element = value
            self.add(self._nested_element)
            self._buffers_in = 0
            self._buffers_out = 0
            self._sink_pad.set_target(self._nested_element.get_static_pad('sink'))
            self._src_pad.set_target(self._nested_element.get_static_pad('src'))
            self._nested_element.sync_state_with_parent()

        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def _on_sink_buffer(self, pad: Gst.Pad, info: Gst.PadProbeInfo):
        self._buffers_in += 1
        self._queue_is_ready.clear()
        while self._buffers_in - self._buffers_out > self._max_queue_size:
            self.logger.debug(
                'Waiting for queued buffers to be processed: %s/%s',
                self._buffers_in,
                self._buffers_out,
            )
            self._queue_is_ready.wait(WAIT_QUEUE_INTERVAL)
        self.logger.debug(
            'Sending buffer to nested element: %s/%s',
            self._buffers_in,
            self._buffers_out,
        )
        return Gst.PadProbeReturn.OK

    def _on_src_buffer(self, pad: Gst.Pad, info: Gst.PadProbeInfo):
        self._buffers_out += 1
        self._queue_is_ready.set()
        self.logger.debug(
            'Buffer processed: %s/%s',
            self._buffers_in,
            self._buffers_out,
        )
        return Gst.PadProbeReturn.OK


# register plugin
GObject.type_register(SyncIoWrapperBin)
__gstelementfactory__ = (
    SyncIoWrapperBin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SyncIoWrapperBin,
)
