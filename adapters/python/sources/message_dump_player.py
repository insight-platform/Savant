import signal
import time
from typing import Optional, Tuple

import msgpack
from savant_rs.utils.serialization import (
    Message,
    load_message_from_bytes,
)
from savant_rs.zmq import BlockingWriter, WriterConfigBuilder

from savant.utils.config import opt_config, strtobool
from savant.utils.config import req_config
from savant.utils.logging import get_logger, init_logging
from savant.utils.welcome import get_starting_message

LOGGER_NAME = 'adapters.message_dump_player'

logger = get_logger(LOGGER_NAME)


class Config:
    """Configuration for the adapter."""

    def __init__(self):
        self.playlist_path = req_config('PLAYLIST_PATH')
        self.zmq_endpoint = req_config('ZMQ_ENDPOINT')
        self.sync_output = opt_config('SYNC_OUTPUT', False, strtobool)


class MessageDumpReader:
    """Reads messages from the message dump file."""

    def __init__(self, file_path: str):
        self._logger = get_logger(f'{LOGGER_NAME}.{self.__class__.__name__}')

        try:
            self._list_file = open(file_path, 'rb')
        except Exception:
            raise ValueError(f'Failed to open a file: {file_path}')

        self._message_dump_file = None
        self._unpacker = None
        self._next_message_dump_file()

    def read(self) -> Optional[Tuple[int, str, Message, bytes]]:
        """Unpack the next message from the file."""

        if not self._unpacker:
            return None
        message = None
        while message is None:
            try:
                message = self._unpacker.unpack()
            except msgpack.OutOfData:
                if not self._next_message_dump_file():
                    return None
            except Exception as e:
                self._logger.error('Failed to unpack message: %s', e)
                raise RuntimeError('Failed to unpack message')

        if not isinstance(message, (Tuple, list)) or len(message) != 4:
            self._logger.error('Invalid message format: %s', message)
            raise RuntimeError('Invalid message format')

        ts, topic, meta, content = message

        return ts, bytes(topic).decode(), load_message_from_bytes(meta), content

    def _next_message_dump_file(self):
        """Open the next message dump file from the list."""

        if self._message_dump_file:
            self._message_dump_file.close()

        file_path = self._list_file.readline()
        if not file_path:
            return False

        try:
            self._message_dump_file = open(file_path.strip(), 'rb')
        except FileNotFoundError:
            self._logger.error('Message dump file not found: %s', file_path)
            raise RuntimeError('Message dump file not found')
        except OSError as e:
            self._logger.error('Failed to open a message dump file [%s]: %s', file_path, e)
            raise RuntimeError('Failed to open a message dump file')
        try:
            self._unpacker = msgpack.Unpacker(self._message_dump_file)
        except Exception as e:
            self._logger.error('Failed to create unpacker for message dump file: %s', e)
            raise RuntimeError('Failed to create unpacker for message dump file')

        return True

    def __del__(self):
        try:
            if self._message_dump_file:
                self._message_dump_file.close()
            if self._list_file:
                self._list_file.close()
        except Exception as e:
            self._logger.error('Failed to clean up the resources: %s', e)


class Player:
    """Receives messages from the dump and sends them to ZeroMQ socket."""

    def __init__(
        self,
        reader: MessageDumpReader,
        config: Config,
    ):
        self._logger = get_logger(f'{LOGGER_NAME}.{self.__class__.__name__}')
        self._reader = reader

        config_builder = WriterConfigBuilder(config.zmq_endpoint)
        self._writer = BlockingWriter(config_builder.build())
        self._sync_output = config.sync_output
        # Last sending time in seconds and last frame timestamp in nanoseconds for synchronization
        self._last_send_time = None
        self._last_ts = None

    def play(self):
        self._writer.start()
        message = self._reader.read()
        while message is not None:
            try:
                ts, topic, meta, content = message
                self._send_message(ts, topic, meta, content)
                message = self._reader.read()
            except Exception as e:
                self._logger.error('Failed to send message: %s', e)
                break
        self._logger.info('There are no more messages to send. Stopping the adapter.')
        self._writer.shutdown()

    def _send_message(self, ts: int, topic: str, meta: Message, content: bytes):
        """Send a message to the sink ZeroMQ socket. Synchronize the sending if needed."""

        self._logger.debug('Sending message to the sink ZeroMQ socket')

        if self._sync_output:
            if self._last_send_time is not None:
                delta = (ts - self._last_ts) / 1.0e9 - (time.time() - self._last_send_time)
                if delta > 0:
                    time.sleep(delta)
                elif delta < 0:
                    self._logger.warning('Message is late by %f seconds', -delta)
            self._last_send_time = time.time()
            self._last_ts = ts

        self._writer.send_message(topic, meta, content)


def main():
    init_logging()
    logger.info(get_starting_message('message dump player adapter'))
    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    try:
        config = Config()
        reader = MessageDumpReader(config.playlist_path)
        player = Player(reader, config)
    except Exception as e:
        logger.error('Failed to start the adapter: %s', e)
        exit(1)

    try:
        player.play()
    except KeyboardInterrupt:
        logger.info('Stopping the adapter')
    except Exception as e:
        logger.error('Adapter failed during execution: %s', e)
        exit(1)


if __name__ == '__main__':
    main()
