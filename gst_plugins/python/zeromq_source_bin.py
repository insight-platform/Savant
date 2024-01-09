"""ZeroMQ src bin."""
from gst_plugins.python.savant_rs_video_decode_bin import (
    SAVANT_RS_VIDEO_DECODE_BIN_PROPERTIES,
    SAVANT_RS_VIDEO_DECODE_BIN_SRC_PAD_TEMPLATE,
)
from gst_plugins.python.zeromq_src import ZEROMQ_SRC_PROPERTIES
from savant.gstreamer import GObject, Gst
from savant.utils.logging import LoggerMixin

# Default values of "queue" element
DEFAULT_INGRESS_QUEUE_LENGTH = 200
DEFAULT_INGRESS_QUEUE_SIZE = 10485760


class ZeroMQSourceBin(LoggerMixin, Gst.Bin):
    """Wrapper for "zeromq_src !

    savant_rs_video_decode_bin".
    """

    GST_PLUGIN_NAME = 'zeromq_source_bin'

    __gstmetadata__ = (
        'ZeroMQ video source bin',
        'Bin/Source',
        'Wrapper for "zeromq_src ! savant_rs_video_decode_bin"',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = SAVANT_RS_VIDEO_DECODE_BIN_SRC_PAD_TEMPLATE

    __gproperties__ = {
        **ZEROMQ_SRC_PROPERTIES,
        **SAVANT_RS_VIDEO_DECODE_BIN_PROPERTIES,
        'ingress-queue-length': (
            int,
            'Length of the ingress queue in frames.',
            'Length of the ingress queue in frames (0 - no limit).',
            0,
            GObject.G_MAXINT,
            DEFAULT_INGRESS_QUEUE_LENGTH,
            GObject.ParamFlags.READWRITE,
        ),
        'ingress-queue-byte-size': (
            int,
            'Size of the ingress queue in bytes.',
            'Size of the ingress queue in bytes (0 - no limit).',
            0,
            GObject.G_MAXINT,
            DEFAULT_INGRESS_QUEUE_SIZE,
            GObject.ParamFlags.READWRITE,
        ),
    }

    __gsignals__ = {'shutdown': (GObject.SignalFlags.RUN_LAST, None, ())}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # properties
        self._ingress_queue_length: int = DEFAULT_INGRESS_QUEUE_LENGTH
        self._ingress_queue_byte_size: int = DEFAULT_INGRESS_QUEUE_SIZE

        self._source: Gst.Element = Gst.ElementFactory.make('zeromq_src')
        self.add(self._source)

        self._queue: Gst.Element = Gst.ElementFactory.make('queue')
        self._queue.set_property('max-size-time', 0)
        self.add(self._queue)

        self._decodebin: Gst.Element = Gst.ElementFactory.make(
            'savant_rs_video_decode_bin'
        )
        self.add(self._decodebin)
        assert self._source.link(self._queue)
        assert self._queue.link(self._decodebin)
        self._decodebin.connect('pad-added', self.on_pad_added)
        self._decodebin.connect('pad-removed', self.on_pad_removed)
        self._decodebin.connect('shutdown', self.on_shutdown)

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the metadata
            required to specify parameters
        """
        if prop.name == 'ingress-queue-length':
            return self._ingress_queue_length
        if prop.name == 'ingress-queue-byte-size':
            return self._ingress_queue_byte_size
        if prop.name in ZEROMQ_SRC_PROPERTIES:
            return self._source.get_property(prop.name)
        if prop.name in SAVANT_RS_VIDEO_DECODE_BIN_PROPERTIES:
            return self._decodebin.get_property(prop.name)
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the metadata
            required to specify parameters
        :param value: new value for param, type dependents on param
        """
        if prop.name == 'ingress-queue-length':
            self._ingress_queue_length = value
            self._queue.set_property('max-size-buffers', value)
        elif prop.name == 'ingress-queue-byte-size':
            self._ingress_queue_byte_size = value
            self._queue.set_property('max-size-bytes', value)
        elif prop.name in ZEROMQ_SRC_PROPERTIES:
            self._source.set_property(prop.name, value)
        elif prop.name in SAVANT_RS_VIDEO_DECODE_BIN_PROPERTIES:
            self._decodebin.set_property(prop.name, value)
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    # pylint: disable=unused-argument
    def on_pad_added(self, element: Gst.Element, pad: Gst.Pad):
        """Proxy newly added pad to bin."""
        ghost_pad: Gst.GhostPad = Gst.GhostPad.new(pad.get_name(), pad)
        ghost_pad.set_active(True)
        self.add_pad(ghost_pad)

    # pylint: disable=unused-argument
    def on_pad_removed(self, element: Gst.Element, pad: Gst.Pad):
        """Remove ghost pad for removed pad."""
        for ghost_pad in self.iterate_pads():
            if ghost_pad.get_name() == pad.get_name():
                self.remove_pad(ghost_pad)
                return

    def on_shutdown(self, element: Gst.Element):
        """Handle shutdown signal."""

        self.logger.info(
            'Received shutdown signal from %s. Passing it downstream.',
            element.get_name(),
        )
        self.emit('shutdown')


# register plugin
GObject.type_register(ZeroMQSourceBin)
__gstelementfactory__ = (
    ZeroMQSourceBin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    ZeroMQSourceBin,
)
