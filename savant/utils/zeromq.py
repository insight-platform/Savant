"""ZeroMQ utilities."""
import logging
from enum import Enum
from typing import Optional, Type, Union

import zmq

logger = logging.getLogger(__name__)


class ZMQException(Exception):
    """Error in ZMQ-related code."""


class ZMQSocketEndpointException(ZMQException):
    """Error in ZMQ socket endpoint."""


class ZMQSocketTypeException(ZMQException):
    """Error in ZMQ socket type."""


class ReceiverSocketTypes(Enum):
    """Receiver socket types."""

    PULL = zmq.PULL
    SUB = zmq.SUB
    REP = zmq.REP


class SenderSocketTypes(Enum):
    """Sender socket types."""

    PUSH = zmq.PUSH
    PUB = zmq.PUB
    REQ = zmq.REQ


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
    """

    def __init__(
        self,
        socket: str,
        socket_type: str = ReceiverSocketTypes.PULL.name,
        bind: bool = True,
        receive_timeout: int = 1000,
    ):
        logger.debug(
            'Initializing ZMQ source: socket %s, type %s, bind %s.',
            socket,
            socket_type,
            bind,
        )

        # might raise exceptions
        # will be handled in ZeromqSrc element
        # or image_files.py / metadata_json.py Python sinks
        socket = get_socket_endpoint(socket)
        self.socket_type = get_socket_type(socket_type, ReceiverSocketTypes)

        self.receive_timeout = receive_timeout
        self.zmq_context = zmq.Context()
        self.receiver = self.zmq_context.socket(self.socket_type.value)
        self.receiver.setsockopt(zmq.RCVHWM, 1)
        if bind:
            self.receiver.bind(socket)
        else:
            self.receiver.connect(socket)
        if self.socket_type == ReceiverSocketTypes.SUB:
            self.receiver.setsockopt_string(zmq.SUBSCRIBE, '')
        self.receiver.setsockopt(zmq.RCVTIMEO, self.receive_timeout)
        self.is_alive = True
        if self.socket_type == ReceiverSocketTypes.REP:
            self._response = b'ok'
        else:
            self._response = None

    def next_message(self) -> Optional[bytes]:
        """Try to receive next message."""
        try:
            message = self.receiver.recv()
        except zmq.Again:
            logger.debug('Timeout exceeded when receiving the next frame')
            return None
        if self._response is not None:
            self.receiver.send(self._response)
        return message

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
