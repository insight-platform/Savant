"""ZeroMQ src bin."""
from savant.gst_plugins.python.avro_video_decode_bin import (
    AVRO_VIDEO_DECODE_BIN_PROPERTIES,
    AVRO_VIDEO_DECODE_BIN_SRC_PAD_TEMPLATE,
)
from savant.gst_plugins.python.zeromq_src import ZEROMQ_SRC_PROPERTIES
from savant.gstreamer import GObject, Gst
from savant.gstreamer.utils import LoggerMixin


class ZeroMQSourceBin(LoggerMixin, Gst.Bin):
    """Wrapper for "zeromq_src !

    avro_video_decode_bin".
    """

    GST_PLUGIN_NAME = 'zeromq_source_bin'

    __gstmetadata__ = (
        'ZeroMQ video source bin',
        'Bin/Source',
        'Wrapper for "zeromq_src ! avro_video_decode_bin"',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = AVRO_VIDEO_DECODE_BIN_SRC_PAD_TEMPLATE

    __gproperties__ = {
        **ZEROMQ_SRC_PROPERTIES,
        **AVRO_VIDEO_DECODE_BIN_PROPERTIES,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._source: Gst.Element = Gst.ElementFactory.make('zeromq_src')
        self.add(self._source)

        self._queue: Gst.Element = Gst.ElementFactory.make('queue')
        self.add(self._queue)

        self._decodebin: Gst.Element = Gst.ElementFactory.make('avro_video_decode_bin')
        self.add(self._decodebin)
        assert self._source.link(self._queue)
        assert self._queue.link(self._decodebin)
        self._decodebin.connect('pad-added', self.on_pad_added)
        self._decodebin.connect('pad-removed', self.on_pad_removed)

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the metadata
            required to specify parameters
        """
        if prop.name in ZEROMQ_SRC_PROPERTIES:
            return self._source.get_property(prop.name)
        if prop.name in AVRO_VIDEO_DECODE_BIN_PROPERTIES:
            return self._decodebin.get_property(prop.name)
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the metadata
            required to specify parameters
        :param value: new value for param, type dependents on param
        """
        if prop.name in ZEROMQ_SRC_PROPERTIES:
            self._source.set_property(prop.name, value)
        elif prop.name in AVRO_VIDEO_DECODE_BIN_PROPERTIES:
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


# register plugin
GObject.type_register(ZeroMQSourceBin)
__gstelementfactory__ = (
    ZeroMQSourceBin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    ZeroMQSourceBin,
)
