"""ZeroMQ utilities."""
import asyncio
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Type, Union
from urllib.parse import urlparse

import zmq
import zmq.asyncio
from cachetools import LRUCache
from savant_rs.zmq import (
    BlockingReader,
    NonBlockingReader,
    ReaderConfig,
    ReaderConfigBuilder,
    ReaderResultEndOfStream,
    ReaderResultMessage,
    ReaderResultPrefixMismatch,
    ReaderResultTimeout,
    ReaderSocketType,
    TopicPrefixSpec,
    WriterSocketType,
)

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

    SUB = ReaderSocketType.Sub
    REP = ReaderSocketType.Rep
    ROUTER = ReaderSocketType.Router


class SenderSocketTypes(Enum):
    """Sender socket types."""

    PUB = WriterSocketType.Pub
    REQ = WriterSocketType.Req
    DEALER = WriterSocketType.Dealer


class Defaults:
    RECEIVE_TIMEOUT = 1000
    SENDER_RECEIVE_TIMEOUT = 5000
    RECEIVE_HWM = 50
    SEND_HWM = 50
    RECEIVE_RETRIES = 3


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


class BaseZeroMQSource(ABC):
    """Base ZeroMQ Source class.

    :param socket: zmq socket endpoint
    :param socket_type: zmq socket type
    :param bind: zmq socket mode (bind or connect)
    :param receive_timeout: receive timeout socket option
    :param receive_hwm: high watermark for inbound messages
    :param source_id: filter inbound messages by source ID
    :param source_id_prefix: filter inbound messages by topic prefix
    """

    receiver: Union[BlockingReader, NonBlockingReader]

    def __init__(
        self,
        socket: str,
        socket_type: str = ReceiverSocketTypes.ROUTER.name,
        bind: bool = True,
        receive_timeout: int = Defaults.RECEIVE_TIMEOUT,
        receive_hwm: int = Defaults.RECEIVE_HWM,
        source_id: Optional[str] = None,
        source_id_prefix: Optional[str] = None,
        routing_ids_cache_size: int = 1000,
        set_ipc_socket_permissions: bool = True,
    ):
        logger.debug(
            'Initializing ZMQ source: socket %s, type %s, bind %s.',
            socket,
            socket_type,
            bind,
        )

        config_builder = ReaderConfigBuilder(socket)
        if not get_zmq_socket_uri_options(socket):
            config_builder.with_socket_type(ReceiverSocketTypes[socket_type].value)
            config_builder.with_bind(bind)
        if source_id:
            config_builder.with_topic_prefix_spec(TopicPrefixSpec.source_id(source_id))
        elif source_id_prefix:
            config_builder.with_topic_prefix_spec(
                TopicPrefixSpec.prefix(source_id_prefix)
            )
        config_builder.with_receive_hwm(receive_hwm)
        config_builder.with_receive_timeout(receive_timeout)

        self.reader = self._create_zmq_reader(config_builder.build())
        self.set_ipc_socket_permissions = set_ipc_socket_permissions
        self.routing_id_filter = RoutingIdFilter(routing_ids_cache_size)

    def start(self):
        """Start ZeroMQ source."""

        if self.is_started:
            logger.warning('ZeroMQ source is already started.')
            return

        logger.info('Starting ZMQ source.')
        self.reader.start()

    @property
    def is_started(self):
        return self.reader.is_started()

    @abstractmethod
    def next_message(self) -> Optional[ReaderResultMessage]:
        """Try to receive next message."""
        pass

    def _filter_result(self, result: ReaderResultMessage) -> bool:
        if isinstance(result, ReaderResultMessage):
            if self.routing_id_filter.filter(result):
                return True
        elif isinstance(result, ReaderResultTimeout):
            logger.debug('Timeout exceeded when receiving the next frame')
        elif isinstance(result, ReaderResultPrefixMismatch):
            logger.debug('Skipping message from topic %s', result.topic)
        elif isinstance(result, ReaderResultEndOfStream):
            logger.debug('Received end of stream message from topic %s', result.topic)

        return False

    def terminate(self):
        """Finish and free zmq socket."""
        if not self.is_started:
            logger.warning('ZeroMQ source is not started.')
            return
        logger.info('Terminating ZeroMQ context.')
        self.reader.shutdown()
        logger.info('ZeroMQ context terminated')

    @abstractmethod
    def _create_zmq_reader(self, config: ReaderConfig):
        pass


class ZeroMQSource(BaseZeroMQSource):
    """ZeroMQ Source class."""

    reader: BlockingReader

    def next_message(self) -> Optional[ReaderResultMessage]:
        """Try to receive next message."""

        if not self.reader.is_started():
            raise RuntimeError('ZeroMQ source is not started.')

        result = self.reader.receive()
        if self._filter_result(result):
            return result

    def __iter__(self):
        return self

    def __next__(self) -> ReaderResultMessage:
        message = None
        while self.reader.is_started() and message is None:
            message = self.next_message()
        if message is None:
            raise StopIteration
        return message

    def _create_zmq_reader(self, config: ReaderConfig):
        return BlockingReader(config)


class AsyncZeroMQSource(ZeroMQSource):
    """Async ZeroMQ Source class."""

    reader: NonBlockingReader

    def _create_zmq_reader(self, config: ReaderConfig):
        return NonBlockingReader(config, 10)  # TODO: make configurable

    async def _try_receive(self, loop):
        return await loop.run_in_executor(None, self.reader.try_receive)

    async def next_message(self) -> Optional[ReaderResultMessage]:
        """Try to receive next message."""

        if not self.reader.is_started():
            raise RuntimeError('ZeroMQ source is not started.')

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.reader.try_receive)
        while result is None:
            await asyncio.sleep(0.01)  # TODO: make configurable
            result = await loop.run_in_executor(None, self.reader.try_receive)

        if self._filter_result(result):
            return result

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = None
        while self.reader.is_started() and message is None:
            message = await self.next_message()
        if message is None:
            raise StopIteration
        return message


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

    def filter(self, message: ReaderResultMessage):
        """Decide whether we need to accept of ignore the message from that routing ID."""

        if not message.routing_id:
            return True

        topic = bytes(message.topic)
        routing_id = bytes(message.routing_id)
        if topic not in self.routing_ids:
            self.routing_ids[topic] = routing_id
            self.routing_ids_cache[(topic, routing_id)] = None

        elif self.routing_ids[topic] != routing_id:
            if (topic, routing_id) in self.routing_ids_cache:
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
                self.routing_ids_cache[(topic, routing_id)] = None

        return True


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


async def async_receive_response(sender: zmq.asyncio.Socket, retries: int):
    """Receive response from async sender socket.

    Retry until response is received.
    """

    while retries > 0:
        try:
            return await sender.recv()
        except zmq.Again:
            retries -= 1
            logger.debug(
                'Timeout exceeded when receiving response (%s retries left)',
                retries,
            )
            if retries == 0:
                raise


def ipc_socket_chmod(socket: str, permission: int = 0o777):
    """Set permissions for IPC socket.

    Needed to make socket available for non-root users.
    """

    parsed = urlparse(socket)
    if parsed.scheme == 'ipc':
        logger.debug('Setting socket permissions to %o (%s).', permission, socket)
        os.chmod(parsed.path, permission)


def create_ipc_socket_dirs(socket: str):
    """Create parent directories for an IPC socket."""

    parsed = urlparse(socket)
    if parsed.scheme == 'ipc':
        dir_name = os.path.dirname(parsed.path)
        if not os.path.exists(dir_name):
            logger.debug(
                'Making directories for ipc socket %s, path %s.', socket, dir_name
            )
            os.makedirs(dir_name)


def get_zmq_socket_uri_options(uri: str) -> Optional[str]:
    socket_options, _ = socket_uri_pattern.fullmatch(uri).groups()
    return socket_options
