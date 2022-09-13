"""ZeroMQ sink."""
import zmq

from savant.gst_plugins.python.zeromq_properties import (
    socket_type_property,
    ZEROMQ_PROPERTIES,
)
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.utils import LoggerMixin
from savant.utils.zeromq import SenderSocketTypes


class ZeroMQSink(LoggerMixin, GstBase.BaseSink):
    """ZeroMQSink GstPlugin."""

    GST_PLUGIN_NAME = 'zeromq_sink'

    __gstmetadata__ = (
        'ZeroMQ sink',
        'Source',
        'Writes binary messages to ZeroMQ',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = Gst.PadTemplate.new(
        'sink',
        Gst.PadDirection.SINK,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.new_any(),
    )

    __gproperties__ = {
        **ZEROMQ_PROPERTIES,
        'socket-type': socket_type_property(SenderSocketTypes),
    }

    def __init__(self):
        GstBase.BaseSink.__init__(self)
        self.socket: str = None
        self.socket_type: SenderSocketTypes = SenderSocketTypes.PUSH
        self.bind: bool = True
        self.zmq_context: zmq.Context = None
        self.sender: zmq.Socket = None
        self.set_sync(False)

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates
            the metadata required to specify parameters
        """
        if prop.name == 'socket':
            return self.socket
        if prop.name == 'socket-type':
            return self.socket_type.name
        if prop.name == 'bind':
            return self.bind
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: structure that encapsulates
            the metadata required to specify parameters
        :param value: new value for param, type dependents on param
        """
        if prop.name == 'socket':
            self.socket = value
        elif prop.name == 'socket-type':
            try:
                self.socket_type = SenderSocketTypes[value]
            except KeyError as exc:
                raise AttributeError(f'Incorrect socket type: {value}') from exc
        elif prop.name == 'bind':
            self.bind = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        """Start source."""
        assert self.socket is not None, '"socket" property is required.'
        self.zmq_context = zmq.Context()
        self.sender = self.zmq_context.socket(self.socket_type.value)
        if self.bind:
            self.sender.bind(self.socket)
        else:
            self.sender.connect(self.socket)
        return True

    def do_render(self, buffer: Gst.Buffer):
        """Send data through ZeroMQ."""
        self.logger.debug(
            'Processing frame %s of size %s', buffer.pts, buffer.get_size()
        )
        mapinfo: Gst.MapInfo
        result, mapinfo = buffer.map(Gst.MapFlags.READ)
        assert result, 'Cannot read buffer.'
        data = mapinfo.data
        self.logger.debug('Sending %s bytes to socket %s.', len(data), self.socket)
        self.sender.send(data)
        buffer.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def do_stop(self):
        """Stop source."""
        self.logger.info('Closing ZeroMQ socket')
        self.sender.close()
        self.logger.info('Terminating ZeroMQ context.')
        self.zmq_context.term()
        self.logger.info('ZeroMQ context terminated')
        return True


# register plugin
GObject.type_register(ZeroMQSink)
__gstelementfactory__ = (
    ZeroMQSink.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    ZeroMQSink,
)
