"""ZeroMQ utilities."""
import logging
from enum import Enum
from typing import Optional

import zmq


class ReceiverSocketTypes(Enum):
    """Receiver socket types."""

    PULL = zmq.PULL
    SUB = zmq.SUB


class SenderSocketTypes(Enum):
    """Sender socket types."""

    PUSH = zmq.PUSH
    PUB = zmq.PUB


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
        socket_type: ReceiverSocketTypes = ReceiverSocketTypes.PULL,
        bind: bool = True,
        receive_timeout: int = 1000,
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.debug(
              'Init ZMQ source, socket %s, type %s, bind %s.', socket, socket_type, bind
        )
        self.receive_timeout = receive_timeout
        self.zmq_context = zmq.Context()
        self.receiver = self.zmq_context.socket(socket_type.value)
        if bind:
            self.receiver.bind(socket)
        else:
            self.receiver.connect(socket)
        if socket_type == ReceiverSocketTypes.SUB:
            self.receiver.setsockopt_string(zmq.SUBSCRIBE, '')
        self.receiver.setsockopt(zmq.RCVTIMEO, self.receive_timeout)
        self.is_alive = True

    def next_message(self) -> Optional[bytes]:
        """Try to receive next message."""
        try:
            return self.receiver.recv()
        except zmq.Again:
            self.logger.debug('Timeout exceeded when receiving the next frame')
            return None

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
        self.logger.info('Closing ZeroMQ socket')
        self.receiver.close()
        self.logger.info('Terminating ZeroMQ context.')
        self.zmq_context.term()
        self.logger.info('ZeroMQ context terminated')
