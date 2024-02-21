"""ZeroMQ utilities."""
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, NamedTuple, Optional, Union

from savant_rs.utils.serialization import Message
from savant_rs.zmq import (
    BlockingReader,
    NonBlockingReader,
    ReaderConfig,
    ReaderConfigBuilder,
    ReaderResultMessage,
    ReaderResultPrefixMismatch,
    ReaderResultTimeout,
    ReaderSocketType,
    TopicPrefixSpec,
    WriterSocketType,
)

from savant.utils.logging import get_logger

from .re_patterns import socket_uri_pattern

logger = get_logger(__name__)


class ZeroMQMessage(NamedTuple):
    topic: List[int]
    message: Message
    content: Optional[bytes] = None


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
        set_ipc_socket_permissions: bool = True,
    ):
        logger.debug(
            'Initializing ZMQ source: socket %s, type %s, bind %s.',
            socket,
            socket_type,
            bind,
        )

        config_builder = ReaderConfigBuilder(socket)
        socket_options = get_zmq_socket_uri_options(socket)
        if socket_options:
            bind = 'bind' in socket_options
        else:
            config_builder.with_socket_type(ReceiverSocketTypes[socket_type].value)
            config_builder.with_bind(bool(bind))  # in case "bind" is "int"
        if source_id:
            config_builder.with_topic_prefix_spec(TopicPrefixSpec.source_id(source_id))
        elif source_id_prefix:
            config_builder.with_topic_prefix_spec(
                TopicPrefixSpec.prefix(source_id_prefix)
            )
        config_builder.with_receive_hwm(receive_hwm)
        config_builder.with_receive_timeout(receive_timeout)
        if bind:
            # IPC permissions can only be set for bind sockets.
            config_builder.with_fix_ipc_permissions(set_ipc_socket_permissions)

        self.reader = self._create_zmq_reader(config_builder.build())

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
    def next_message(self) -> Optional[ZeroMQMessage]:
        """Try to receive next message."""
        pass

    def _build_result(self, result: ReaderResultMessage) -> Optional[ZeroMQMessage]:
        if isinstance(result, ReaderResultMessage):
            return ZeroMQMessage(
                result.topic,
                result.message,
                b''.join(result.data(i) for i in range(result.data_len())),
            )
        elif isinstance(result, ReaderResultTimeout):
            logger.debug('Timeout exceeded when receiving the next frame')
        elif isinstance(result, ReaderResultPrefixMismatch):
            logger.debug('Skipping message from topic %s', result.topic)

        return None

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

    def next_message(self) -> Optional[ZeroMQMessage]:
        """Try to receive next message."""

        if not self.reader.is_started():
            raise RuntimeError('ZeroMQ source is not started.')

        result = self.reader.receive()
        return self._build_result(result)

    def __iter__(self):
        return self

    def __next__(self) -> ZeroMQMessage:
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

    async def next_message(self) -> Optional[ZeroMQMessage]:
        """Try to receive next message."""

        if not self.reader.is_started():
            raise RuntimeError('ZeroMQ source is not started.')

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.reader.try_receive)
        while result is None:
            await asyncio.sleep(0.01)  # TODO: make configurable
            result = await loop.run_in_executor(None, self.reader.try_receive)

        return self._build_result(result)

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = None
        while self.reader.is_started() and message is None:
            message = await self.next_message()
        if message is None:
            raise StopIteration
        return message


def get_zmq_socket_uri_options(uri: str) -> Optional[str]:
    socket_options, _ = socket_uri_pattern.fullmatch(uri).groups()
    return socket_options
