"""ZeroMQ utilities."""
import logging
from enum import Enum
from typing import Optional, Type, Union

import zmq
from cachetools import LRUCache

logger = logging.getLogger(__name__)


class ZMQException(Exception):
    """Error in ZMQ-related code."""


class ZMQSocketEndpointException(ZMQException):
    """Error in ZMQ socket endpoint."""


class ZMQSocketTypeException(ZMQException):
    """Error in ZMQ socket type."""


class ReceiverSocketTypes(Enum):
    """Receiver socket types."""

    SUB = zmq.SUB
    REP = zmq.REP
    ROUTER = zmq.ROUTER


class SenderSocketTypes(Enum):
    """Sender socket types."""

    PUB = zmq.PUB
    REQ = zmq.REQ
    DEALER = zmq.DEALER


class Defaults:
    RECEIVE_TIMEOUT = 1000
    RECEIVE_HWM = 50
    SEND_HWM = 50


def get_socket_endpoint(socket_endpoint: str):
    if not isinstance(socket_endpoint, str):
        raise ZMQSocketEndpointException(
            f'Incorrect socket endpoint: "{socket_endpoint}":'
            f'"{type(socket_endpoint)}" is not string.'
        )
    return socket_endpoint


def get_socket_type(
    socket_type_name: str,
    socket_type_enum: Union[Type[ReceiverSocketTypes], Type[SenderSocketTypes]],
):
    if not isinstance(socket_type_name, str):
        raise ZMQSocketTypeException(
            f'Incorrect socket_type_name: "{socket_type_name}":'
            f'"{type(socket_type_name)}" is not string.'
        )

    socket_type_name = str.upper(socket_type_name)

    try:
        return socket_type_enum[socket_type_name]
    except KeyError as exc:
        raise ZMQSocketTypeException(
            f'Incorrect socket type: {socket_type_name} is not one of '
            f'{[socket_type.name for socket_type in socket_type_enum]}.'
        ) from exc


class ZeroMQSource:
    """ZeroMQ Source class.

    :param socket: zmq socket endpoint
    :param socket_type: zmq socket type
    :param bind: zmq socket mode (bind or connect)
    :param receive_timeout: receive timeout socket option
    :param receive_hwm: high watermark for inbound messages
    :param topic_prefix: filter inbound messages by topic prefix
    """

    def __init__(
        self,
        socket: str,
        socket_type: str = ReceiverSocketTypes.ROUTER.name,
        bind: bool = True,
        receive_timeout: int = Defaults.RECEIVE_TIMEOUT,
        receive_hwm: int = Defaults.RECEIVE_HWM,
        topic_prefix: Optional[str] = None,
        routing_ids_cache_size: int = 1000,
    ):
        logger.debug(
            'Initializing ZMQ source: socket %s, type %s, bind %s.',
            socket,
            socket_type,
            bind,
        )

        self.topic_prefix = topic_prefix.encode() if topic_prefix else b''

        # might raise exceptions
        # will be handled in ZeromqSrc element
        # or image_files.py / metadata_json.py Python sinks
        socket = get_socket_endpoint(socket)
        self.socket_type = get_socket_type(socket_type, ReceiverSocketTypes)
        self.message_len = 2  # source_id, data

        self.receive_timeout = receive_timeout
        self.zmq_context = zmq.Context()
        self.receiver = self.zmq_context.socket(self.socket_type.value)
        self.receiver.setsockopt(zmq.RCVHWM, receive_hwm)
        if bind:
            self.receiver.bind(socket)
        else:
            self.receiver.connect(socket)
        if self.socket_type == ReceiverSocketTypes.SUB:
            self.receiver.setsockopt(zmq.SUBSCRIBE, self.topic_prefix)
        elif self.socket_type == ReceiverSocketTypes.ROUTER:
            self.message_len += 1  # routing_id, ...
        self.routing_id_filter = RoutingIdFilter(routing_ids_cache_size)
        self.receiver.setsockopt(zmq.RCVTIMEO, self.receive_timeout)
        self.is_alive = True
        if self.socket_type == ReceiverSocketTypes.REP:
            self._response = b'ok'
        else:
            self._response = None

    def next_message(self) -> Optional[bytes]:
        """Try to receive next message."""
        try:
            message = self.receiver.recv_multipart()
        except zmq.Again:
            logger.debug('Timeout exceeded when receiving the next frame')
            return
        if not len(message) == self.message_len:
            raise RuntimeError(
                f'Invalid number of ZeroMQ message parts: got {len(message)} expected {self.message_len}.'
            )
        if self._response is not None:
            self.receiver.send(self._response)
        if self.socket_type == ReceiverSocketTypes.ROUTER:
            routing_id, topic, data = message
        else:
            topic, data = message
            routing_id = None

        if self.topic_prefix and not topic.startswith(self.topic_prefix):
            logger.debug(
                'Skipping message from topic %s, expected prefix %s',
                topic,
                self.topic_prefix,
            )
            return

        if self.routing_id_filter.filter(routing_id, topic):
            return data

    def __iter__(self):
        return self

    def __next__(self):
        message = None
        while self.is_alive and message is None:
            message = self.next_message()
        if message is None:
            raise StopIteration
        return message

    def terminate(self):
        """Finish and free zmq socket."""
        self.is_alive = False
        logger.info('Closing ZeroMQ socket')
        self.receiver.close()
        logger.info('Terminating ZeroMQ context.')
        self.zmq_context.term()
        logger.info('ZeroMQ context terminated')


class RoutingIdFilter:
    """Cache for routing IDs to filter out old connections.

    Some ZeroMQ sockets have buffer on the receiver side (PUSH/PULL, DEALER/ROUTER).
    ZeroMQ processes messages in round-robin manner. When a sender reconnects
    with the same source ID ZeroMQ mixes up messages from old and new connections.
    This causes decoder to fail and the module freezes. To avoid this we are
    caching all routing IDs and ignoring messages from the old ones.
    """

    def __init__(self, cache_size: int):
        self.routing_ids = {}
        self.routing_ids_cache = LRUCache(cache_size)

    def filter(self, routing_id: Optional[bytes], topic: bytes):
        """Decide whether we need to accept of ignore the message from that routing ID."""

        if not routing_id:
            return True

        if topic not in self.routing_ids:
            self.routing_ids[topic] = routing_id
            self.routing_ids_cache[routing_id] = None

        elif self.routing_ids[topic] != routing_id:
            if routing_id in self.routing_ids_cache:
                logger.debug(
                    'Skipping message from topic %s: routing ID %s, expected %s.',
                    topic,
                    routing_id,
                    self.routing_ids[topic],
                )
                return False

            else:
                logger.debug(
                    'Routing ID for topic %s changed from %s to %s.',
                    topic,
                    self.routing_ids[topic],
                    routing_id,
                )
                self.routing_ids[topic] = routing_id
                self.routing_ids_cache[routing_id] = None

        return True


def build_topic_prefix(
    source_id: Optional[str],
    source_id_prefix: Optional[str],
) -> Optional[str]:
    """Build topic prefix based on source ID or its prefix."""
    if source_id:
        return f'{source_id}/'
    elif source_id_prefix:
        return source_id_prefix
