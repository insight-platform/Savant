from typing import Optional

from savant_rs.zmq import BlockingWriter, WriterConfigBuilder

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
        writer_config_builder = WriterConfigBuilder(output_socket)
        writer_config_builder.with_socket_type(SenderSocketTypes.PUB.value)
        writer_config_builder.with_bind(True)
        writer_config_builder.with_send_hwm(Defaults.SEND_HWM)
        self.writer_config = writer_config_builder.build()
        self.sender: Optional[BlockingWriter] = None

    def start(self):
        self.sender = BlockingWriter(self.writer_config)
        self.source.start()

    def run(self):
        while True:
            message = self.source.next_message()
            if message is not None:
                self.sender.send_message(message.topic, message.message)
