"""ZeroMQ sink."""
import inspect
import zmq

from savant.gst_plugins.python.zeromq_properties import (
    socket_type_property,
    ZEROMQ_PROPERTIES,
)
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.utils import LoggerMixin, propagate_gst_setting_error
from savant.utils.zeromq import (
    Defaults,
    SenderSocketTypes,
    ZMQException,
    get_socket_type,
    get_socket_endpoint,
)


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
        'send-hwm': (
            int,
            'High watermark for outbound messages',
            'High watermark for outbound messages',
            1,
            GObject.G_MAXINT,
            Defaults.SEND_HWM,
            GObject.ParamFlags.READWRITE,
        ),
        'source-id': (
            str,
            'Source ID',
            'Source ID, e.g. "camera1".',
            None,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        GstBase.BaseSink.__init__(self)
        self.socket: str = None
        self.socket_type: SenderSocketTypes = SenderSocketTypes.REQ
        self.bind: bool = True
        self.zmq_context: zmq.Context = None
        self.sender: zmq.Socket = None
        self.wait_response = False
        self.send_hwm = Defaults.SEND_HWM
        self.source_id: str = None
        self.zmq_topic: bytes = None
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
        if prop.name == 'send-hwm':
            return self.send_hwm
        if prop.name == 'source-id':
            return self.source_id
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: structure that encapsulates
            the metadata required to specify parameters
        :param value: new value for param, type dependents on param
        """
        self.logger.debug('Setting property "%s" to "%s".', prop.name, value)
        if prop.name == 'socket':
            self.socket = value
        elif prop.name == 'socket-type':
            self.socket_type = value
        elif prop.name == 'bind':
            self.bind = value
        elif prop.name == 'send-hwm':
            self.send_hwm = value
        elif prop.name == 'source-id':
            self.source_id = value
            self.zmq_topic = f'{value}/'.encode()
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        """Start source."""
        assert self.source_id, 'Source ID is required.'
        try:
            self.socket = get_socket_endpoint(self.socket)
            self.socket_type = get_socket_type(self.socket_type, SenderSocketTypes)
        except ZMQException:
            self.logger.exception('Element start error.')
            frame = inspect.currentframe()
            propagate_gst_setting_error(self, frame, __file__)
            # prevents pipeline from starting
            return False

        self.wait_response = self.socket_type == SenderSocketTypes.REQ
        self.zmq_context = zmq.Context()
        self.sender = self.zmq_context.socket(self.socket_type.value)
        self.sender.setsockopt(zmq.SNDHWM, self.send_hwm)
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
        self.sender.send_multipart([self.zmq_topic, data])
        if self.wait_response:
            resp = self.sender.recv()
            self.logger.debug(
                'Received %s bytes from socket %s.', len(resp), self.socket
            )
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
