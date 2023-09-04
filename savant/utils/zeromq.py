"""ZeroMQ utilities."""
import os
from enum import Enum
from typing import List, Optional, Tuple, Type, Union
from urllib.parse import urlparse

import zmq
from cachetools import LRUCache

from savant.utils.logging import get_logger

from .re_patterns import socket_options_pattern, socket_uri_pattern

logger = get_logger(__name__)

CONFIRMATION_MESSAGE = b'OK'
END_OF_STREAM_MESSAGE = b'EOS'


class ZMQException(Exception):
    """Error in ZMQ-related code."""


class ZMQSocketEndpointException(ZMQException):
    """Error in ZMQ socket endpoint."""


class ZMQSocketTypeException(ZMQException):
    """Error in ZMQ socket type."""


class ZMQSocketUriParsingException(ZMQException):
    """Error in ZMQ socket URI."""


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
    SENDER_RECEIVE_TIMEOUT = 5000
    RECEIVE_HWM = 50
    SEND_HWM = 50
    REQ_RECEIVE_RETRIES = 3
    EOS_CONFIRMATION_RETRIES = 3


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
        set_ipc_socket_permissions: bool = True,
    ):
        logger.debug(
            'Initializing ZMQ source: socket %s, type %s, bind %s.',
            socket,
            socket_type,
            bind,
        )

        self.topic_prefix = topic_prefix.encode() if topic_prefix else b''
        self.receive_hwm = receive_hwm
        self.set_ipc_socket_permissions = set_ipc_socket_permissions

        # might raise exceptions
        # will be handled in ZeromqSrc element
        # or image_files.py / metadata_json.py Python sinks
        self.socket_type, self.bind, self.socket = parse_zmq_socket_uri(
            uri=socket,
            socket_type_name=socket_type,
            socket_type_enum=ReceiverSocketTypes,
            bind=bind,
        )

        self.receive_timeout = receive_timeout
        self.zmq_context: Optional[zmq.Context] = None
        self.receiver: Optional[zmq.Socket] = None
        self.routing_id_filter = RoutingIdFilter(routing_ids_cache_size)
        self.is_alive = False
        self._always_respond = self.socket_type == ReceiverSocketTypes.REP

    def start(self):
        """Start ZeroMQ source."""

        if self.is_alive:
            logger.warning('ZeroMQ source is already started.')
            return

        logger.info(
            'Starting ZMQ source: socket %s, type %s, bind %s.',
            self.socket,
            self.socket_type,
            self.bind,
        )

        self.zmq_context = zmq.Context()
        self.receiver = self.zmq_context.socket(self.socket_type.value)
        self.receiver.setsockopt(zmq.RCVHWM, self.receive_hwm)
        if self.bind:
            self.receiver.bind(self.socket)
        else:
            self.receiver.connect(self.socket)
        if self.socket_type == ReceiverSocketTypes.SUB:
            self.receiver.setsockopt(zmq.SUBSCRIBE, self.topic_prefix)
        self.receiver.setsockopt(zmq.RCVTIMEO, self.receive_timeout)
        if self.set_ipc_socket_permissions:
            set_ipc_socket_permissions(self.socket)
        self.is_alive = True

    def next_message_without_routing_id(self) -> Optional[List[bytes]]:
        """Try to receive next message without routing ID but with topic."""
        if not self.is_alive:
            raise RuntimeError('ZeroMQ source is not started.')

        try:
            message = self.receiver.recv_multipart()
        except zmq.Again:
            logger.debug('Timeout exceeded when receiving the next frame')
            return

        if self.socket_type == ReceiverSocketTypes.ROUTER:
            routing_id, *message = message
        else:
            routing_id = None

        if message[0] == END_OF_STREAM_MESSAGE:
            if routing_id:
                self.receiver.send_multipart([routing_id, CONFIRMATION_MESSAGE])
            else:
                self.receiver.send(CONFIRMATION_MESSAGE)
            return

        if self._always_respond:
            self.receiver.send(CONFIRMATION_MESSAGE)

        topic = message[0]
        if len(message) < 2:
            raise RuntimeError(f'ZeroMQ message from topic {topic} does not have data.')

        if self.topic_prefix and not topic.startswith(self.topic_prefix):
            logger.debug(
                'Skipping message from topic %s, expected prefix %s',
                topic,
                self.topic_prefix,
            )
            return

        if self.routing_id_filter.filter(routing_id, topic):
            return message

    def next_message(self) -> Optional[List[bytes]]:
        """Try to receive next message."""
        message = self.next_message_without_routing_id()
        if message is not None:
            return message[1:]

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
        if not self.is_alive:
            logger.warning('ZeroMQ source is not started.')
            return
        self.is_alive = False
        logger.info('Closing ZeroMQ socket')
        self.receiver.close()
        self.receiver = None
        logger.info('Terminating ZeroMQ context.')
        self.zmq_context.term()
        self.zmq_context = None
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


def parse_zmq_socket_uri(
    uri: str,
    socket_type_name: Optional[str],
    socket_type_enum: Union[Type[ReceiverSocketTypes], Type[SenderSocketTypes]],
    bind: Optional[bool],
) -> Tuple[Union[ReceiverSocketTypes, SenderSocketTypes], bool, str]:
    """Parse ZMQ socket URI.

    Socket type and binding flag can be embedded into URI or passed as separate arguments.

    URI schema: [<socket_type>+(bind|connect):]<endpoint>.

    Examples:
        - ipc:///tmp/zmq-sockets/input-video.ipc
        - dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc:source
        - tcp://1.1.1.1:3333
        - pub+bind:tcp://1.1.1.1:3333:source

    :param uri: ZMQ socket URI.
    :param socket_type_name: Name of a socket type. Ignored when specified in URI.
    :param socket_type_enum: Enum for a socket type.
    :param bind: Whether to bind or connect ZMQ socket. Ignored when in URI.
    """

    options, endpoint = socket_uri_pattern.fullmatch(uri).groups()
    if options:
        socket_type_name, bind_str = socket_options_pattern.fullmatch(options).groups()
        if bind_str == 'bind':
            bind = True
        elif bind_str == 'connect':
            bind = False
        else:
            raise ZMQSocketUriParsingException(
                f'Incorrect socket bind options in socket URI {uri!r}'
            )
    if socket_type_name is None:
        raise ZMQSocketUriParsingException(
            f'Socket type is not specified for URI {uri!r}'
        )
    if bind is None:
        raise ZMQSocketUriParsingException(
            f'Socket binding flag is not specified for URI {uri!r}'
        )

    endpoint = get_socket_endpoint(endpoint)
    socket_type = get_socket_type(socket_type_name, socket_type_enum)

    return socket_type, bind, endpoint


def receive_response(sender: zmq.Socket, retries: int):
    """Receive response from sender socket.

    Retry until response is received.
    """

    while retries > 0:
        try:
            return sender.recv()
        except zmq.Again:
            retries -= 1
            logger.debug(
                'Timeout exceeded when receiving response (%s retries left)',
                retries,
            )
            if retries == 0:
                raise


def set_ipc_socket_permissions(socket: str, permission: int = 0o777):
    """Set permissions for IPC socket.

    Needed to make socket available for non-root users.
    """

    parsed = urlparse(socket)
    if parsed.scheme == 'ipc':
        logger.debug('Setting socket permissions to %o (%s).', permission, socket)
        os.chmod(parsed.path, permission)
