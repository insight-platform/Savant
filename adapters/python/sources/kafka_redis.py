import os
import signal
from distutils.util import strtobool
from queue import Queue
from threading import Thread
from typing import Dict, Iterator, Optional, Tuple, Union

from confluent_kafka import Consumer
from redis import Redis
from savant_rs.primitives import EndOfStream, VideoFrame, VideoFrameContent
from savant_rs.utils.serialization import Message, load_message_from_bytes

from savant.api.enums import ExternalFrameType
from savant.client import SourceBuilder
from savant.utils.logging import get_logger, init_logging

LOGGER_NAME = 'adapters.kafka_redis_source'
logger = get_logger(LOGGER_NAME)


def opt_config(name, default=None, convert=None):
    conf_str = os.environ.get(name)
    if conf_str:
        return convert(conf_str) if convert else conf_str
    return default


class Config:
    def __init__(self):
        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.queue_size = opt_config('QUEUE_SIZE', 100, int)
        self.kafka = KafkaConfig()


class KafkaConfig:
    def __init__(self):
        self.brokers = os.environ['KAFKA_BROKERS']
        self.topic = os.environ['KAFKA_TOPIC']
        self.group_id = os.environ['KAFKA_GROUP_ID']
        self.poll_timeout = opt_config('KAFKA_POLL_TIMEOUT', 1, float)
        self.auto_commit = opt_config('KAFKA_AUTO_COMMIT', True, strtobool)
        self.auto_offset_reset = opt_config('KAFKA_AUTO_OFFSET_RESET', 'latest')
        self.partition_assignment_strategy = opt_config(
            'KAFKA_PARTITION_ASSIGNMENT_STRATEGY', 'roundrobin'
        )
        self.max_poll_interval_ms = opt_config(
            'KAFKA_MAX_POLL_INTERVAL_MS', 600000, int
        )


class KafkaRedisSource:
    def __init__(self, config: Config):
        self._config = config
        self._source = SourceBuilder().with_socket(config.zmq_endpoint).build()
        self._consumer = build_consumer(config.kafka)
        self._frame_clients: Dict[str, Redis] = {}
        self._consumer_queue = Queue(config.queue_size)
        self._is_running = False

    def run(self):
        logger.info('Starting the source')
        self._is_running = True
        self._consumer.subscribe([self._config.kafka.topic])
        logger.info('Subscribed to topic %r', self._config.kafka.topic)
        poller_thread = Thread(target=self.poller, name='poller')
        poller_thread.start()
        sender_thread = Thread(target=self.sender, name='sender')
        sender_thread.start()
        try:
            sender_thread.join()
        except KeyboardInterrupt:
            pass
        logger.info('Stopping the source')
        self._is_running = False
        poller_thread.join()
        sender_thread.join()

    def poller(self):
        logger.info('Starting poller')
        while self._is_running:
            message = self._consumer.poll(self._config.kafka.poll_timeout)
            if message is None:
                continue

            data = message.value()
            logger.debug(
                'Polled kafka message %s/%s/%s with key %s. Message size is %s bytes.',
                message.topic(),
                message.partition(),
                message.offset(),
                message.key(),
                len(data),
            )
            self._consumer_queue.put(data)

    def sender(self):
        logger.info('Starting sender')
        for result in self._source.send_iter(self.video_frame_iterator()):
            logger.debug(
                'Status of sending frame for source %s: %s.',
                result.source_id,
                result.status,
            )

    def deserialize_message(
        self,
        data: bytes,
    ) -> Optional[Union[Tuple[VideoFrame, bytes], EndOfStream]]:
        message: Message = load_message_from_bytes(data)
        if message.is_video_frame():
            video_frame = message.as_video_frame()
            return self.fetch_video_frame_content(video_frame)
        if message.is_end_of_stream():
            return message.as_end_of_stream()

        logger.warning('Unsupported message type for message %r', message)
        return None

    def video_frame_iterator(
        self,
    ) -> Iterator[Union[Tuple[VideoFrame, bytes], EndOfStream]]:
        while self._is_running or not self._consumer_queue.empty():
            data = self._consumer_queue.get()
            deserialized = self.deserialize_message(data)
            if deserialized is not None:
                yield deserialized

    def fetch_video_frame_content(
        self,
        video_frame: VideoFrame,
    ) -> Optional[Tuple[VideoFrame, bytes]]:
        if video_frame.content.is_internal():
            content = video_frame.content.get_data_as_bytes()
            video_frame.content = VideoFrameContent.external(
                ExternalFrameType.ZEROMQ.value, None
            )
            return video_frame, content

        if video_frame.content.is_external():
            if video_frame.content.get_method() != ExternalFrameType.REDIS.value:
                logger.warning(
                    'Unsupported external frame type %r',
                    video_frame.content.get_method(),
                )
                return None
            content = self.fetch_content_from_redis(video_frame.content.get_location())
            if content is None:
                logger.warning(
                    'Failed to fetch frame from %r',
                    video_frame.content.get_location(),
                )
                return None

            return video_frame, content

        logger.warning('Unsupported frame content %r', video_frame.content)
        return None

    def fetch_content_from_redis(self, location: str) -> Optional[bytes]:
        logger.debug('Fetching frame from %r', location)
        host_port, key = location.split('/', 1)
        frame_client = self._frame_clients.get(host_port)

        if frame_client is None:
            logger.info('Connecting to %r', host_port)
            host, port = host_port.split(':')
            frame_client = Redis(host=host, port=int(port))
            self._frame_clients[host_port] = frame_client

        return frame_client.get(key)


def build_consumer(config: KafkaConfig) -> Consumer:
    return Consumer(
        {
            'bootstrap.servers': config.brokers,
            'group.id': config.group_id,
            'auto.offset.reset': config.auto_offset_reset,
            'enable.auto.commit': config.auto_commit,
            'partition.assignment.strategy': config.partition_assignment_strategy,
            'max.poll.interval.ms': config.max_poll_interval_ms,
        }
    )


def main():
    init_logging()
    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))
    source = KafkaRedisSource(Config())
    source.run()


if __name__ == '__main__':
    main()
