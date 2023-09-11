from typing import Optional

import zmq

from savant.utils.zeromq import Defaults, SenderSocketTypes, ZeroMQSource


class ZeroMqProxy:
    """A proxy that receives messages from a ZeroMQ socket and forwards them
    to another PUB ZeroMQ socket. Needed for multi-stream Always-On-RTSP sink.
    """

    def __init__(
        self,
        input_socket: str,
        input_socket_type: Optional[str],
        input_bind: Optional[bool],
        output_socket: str,
    ):
        self.source = ZeroMQSource(
            socket=input_socket,
            socket_type=input_socket_type,
            bind=input_bind,
        )
        self.output_socket = output_socket
        self.sender: Optional[zmq.Socket] = None
        self.output_zmq_context: Optional[zmq.Context] = None

    def start(self):
        self.output_zmq_context = zmq.Context()
        self.sender = self.output_zmq_context.socket(SenderSocketTypes.PUB.value)
        self.sender.setsockopt(zmq.SNDHWM, Defaults.SEND_HWM)
        self.sender.bind(self.output_socket)
        self.source.start()

    def run(self):
        while True:
            message = self.source.next_message_without_routing_id()
            if message is not None:
                self.sender.send_multipart(message)
