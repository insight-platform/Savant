"""ZeroMQ src."""
import inspect
from typing import Optional

import zmq

from gst_plugins.python.zeromq_properties import (
    socket_type_property,
    ZEROMQ_PROPERTIES,
)
from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.utils import (
    gst_buffer_from_list,
    LoggerMixin,
    propagate_gst_setting_error,
)
from savant.utils.zeromq import (
    Defaults,
    ReceiverSocketTypes,
    ZeroMQSource,
    ZMQException,
    build_topic_prefix,
)

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
        self.socket: str = None
        self.socket_type: str = ReceiverSocketTypes.ROUTER.name
        self.bind: bool = True
        self.zmq_context: zmq.Context = None
        self.context = None
        self.receiver = None
        self.receive_timeout: int = Defaults.RECEIVE_TIMEOUT
        self.receive_hwm: int = Defaults.RECEIVE_HWM
        self.source_id: Optional[str] = None
        self.source_id_prefix: Optional[str] = None
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
        # if prop.name == 'zmq-context':
        #     return self.zmq_context
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
        # elif prop.name == 'zmq-context':
        #     self.zmq_context = value
        else:
            raise AttributeError(f'Unknown property "{prop.name}".')

    def do_start(self):
        """Start source."""
        self.logger.debug('Called do_start().')
        topic_prefix = build_topic_prefix(self.source_id, self.source_id_prefix)

        try:
            self.source = ZeroMQSource(
                socket=self.socket,
                socket_type=self.socket_type,
                bind=self.bind,
                receive_timeout=self.receive_timeout,
                receive_hwm=self.receive_hwm,
                topic_prefix=topic_prefix,
            )
        except ZMQException:
            self.logger.exception('Element start error.')
            frame = inspect.currentframe()
            propagate_gst_setting_error(self, frame, __file__)
            # prevents pipeline from starting
            return False

        return True

    # pylint: disable=unused-argument
    def do_create(self, offset: int, size: int, buffer: Gst.Buffer = None):
        """Create gst buffer."""

        self.logger.debug('Receiving next message')

        message = None
        while message is None:
            flow_return = self.wait_playing()
            if flow_return != Gst.FlowReturn.OK:
                self.logger.info('Returning %s', flow_return)
                return flow_return, None
            message = self.source.next_message()
        self.logger.debug('Received message of sizes %s', [len(x) for x in message])
        buffer = gst_buffer_from_list(message)

        return Gst.FlowReturn.OK, buffer

    def do_stop(self):
        """Gst src stop callback."""
        self.source.terminate()
        return True

    def do_is_seekable(self):
        """Check if the source can seek."""
        return False


# register plugin
GObject.type_register(ZeromqSrc)
__gstelementfactory__ = (
    ZeromqSrc.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    ZeromqSrc,
)
