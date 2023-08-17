"""ZeroMQ sink."""
import inspect
from typing import List, Union

import zmq

from gst_plugins.python.zeromq_properties import (ZEROMQ_PROPERTIES,
                                                  socket_type_property)
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.utils import (propagate_gst_error,
                                    propagate_gst_setting_error)
from savant.utils.logging import LoggerMixin
from savant.utils.zeromq import (END_OF_STREAM_MESSAGE, Defaults,
                                 SenderSocketTypes, ZMQException,
                                 parse_zmq_socket_uri, receive_response)


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
        'receive-timeout': (
            int,
            'Receive timeout socket option',
            'Receive timeout socket option',
            0,
            GObject.G_MAXINT,
            Defaults.SENDER_RECEIVE_TIMEOUT,
            GObject.ParamFlags.READWRITE,
        ),
        'req-receive-retries': (
            int,
            'Retries to receive confirmation message from REQ socket',
            'Retries to receive confirmation message from REQ socket',
            1,
            GObject.G_MAXINT,
            Defaults.REQ_RECEIVE_RETRIES,
            GObject.ParamFlags.READWRITE,
        ),
        'eos-confirmation-retries': (
            int,
            'Retries to receive EOS confirmation message',
            'Retries to receive EOS confirmation message',
            1,
            GObject.G_MAXINT,
            Defaults.EOS_CONFIRMATION_RETRIES,
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
        self.socket_type: Union[str, SenderSocketTypes] = SenderSocketTypes.DEALER
        self.bind: bool = True
        self.zmq_context: zmq.Context = None
        self.sender: zmq.Socket = None
        self.wait_response = False
        self.send_hwm = Defaults.SEND_HWM
        self.receive_timeout = Defaults.SENDER_RECEIVE_TIMEOUT
        self.req_receive_retries = Defaults.REQ_RECEIVE_RETRIES
        self.eos_confirmation_retries = Defaults.EOS_CONFIRMATION_RETRIES
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
            return (
                self.socket_type.name
                if isinstance(self.socket_type, SenderSocketTypes)
                else self.socket_type
            )
        if prop.name == 'bind':
            return self.bind
        if prop.name == 'send-hwm':
            return self.send_hwm
        if prop.name == 'source-id':
            return self.source_id
        if prop.name == 'receive-timeout':
            return self.receive_timeout
        if prop.name == 'req-receive-retries':
            return self.req_receive_retries
        if prop.name == 'eos-confirmation-retries':
            return self.eos_confirmation_retries
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
        elif prop.name == 'receive-timeout':
            self.receive_timeout = value
        elif prop.name == 'req-receive-retries':
            self.req_receive_retries = value
        elif prop.name == 'eos-confirmation-retries':
            self.eos_confirmation_retries = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        """Start source."""
        assert self.source_id, 'Source ID is required.'
        try:
            self.socket_type, self.bind, self.socket = parse_zmq_socket_uri(
                uri=self.socket,
                socket_type_name=self.socket_type,
                socket_type_enum=SenderSocketTypes,
                bind=self.bind,
            )
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
        self.sender.setsockopt(zmq.RCVTIMEO, self.receive_timeout)
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
        message: List[bytes] = [self.zmq_topic]
        mapinfo_list: List[Gst.MapInfo] = []
        mapinfo: Gst.MapInfo
        result, mapinfo = buffer.map_range(0, 1, Gst.MapFlags.READ)
        assert result, 'Cannot read buffer.'
        mapinfo_list.append(mapinfo)
        message.append(mapinfo.data)
        if buffer.n_memory() > 1:
            # TODO: Use Gst.Meta to check where to split buffer to ZeroMQ message parts
            result, mapinfo = buffer.map_range(1, -1, Gst.MapFlags.READ)
            assert result, 'Cannot read buffer.'
            mapinfo_list.append(mapinfo)
            message.append(mapinfo.data)
        self.logger.debug(
            'Sending %s bytes to socket %s.', sum(len(x) for x in message), self.socket
        )
        self.sender.send_multipart(message)
        if self.wait_response:
            try:
                resp = receive_response(self.sender, self.req_receive_retries)
            except zmq.Again:
                error = (
                    f"The REP socket hasn't responded in a configured timeframe "
                    f"{self.receive_timeout * self.req_receive_retries} ms."
                )
                self.logger.error(error)
                frame = inspect.currentframe()
                propagate_gst_error(
                    gst_element=self,
                    frame=frame,
                    file_path=__file__,
                    domain=Gst.StreamError.quark(),
                    code=Gst.StreamError.FAILED,
                    text=error,
                )
                return Gst.FlowReturn.ERROR

            self.logger.debug(
                'Received %s bytes from socket %s.', len(resp), self.socket
            )
        for mapinfo in mapinfo_list:
            buffer.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def do_stop(self):
        """Stop source."""

        if self.socket_type == SenderSocketTypes.DEALER:
            self.logger.info('Sending End-of-Stream message to socket %s', self.socket)
            self.sender.send_multipart([END_OF_STREAM_MESSAGE])
            self.logger.info(
                'Waiting for End-of-Stream message confirmation from socket %s',
                self.socket,
            )
            try:
                self.sender.recv()
            except zmq.Again:
                error = (
                    f'Timeout exceeded when receiving End-of-Stream message '
                    f'confirmation from socket {self.socket}'
                )
                self.logger.error(error)
                frame = inspect.currentframe()
                propagate_gst_error(
                    gst_element=self,
                    frame=frame,
                    file_path=__file__,
                    domain=Gst.StreamError.quark(),
                    code=Gst.StreamError.FAILED,
                    text=error,
                )
                return False

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
