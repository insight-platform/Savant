import asyncio
import os
from asyncio import Queue
from distutils.util import strtobool
from typing import AsyncIterator, Dict, Optional, Tuple, Union

from confluent_kafka import Consumer
from redis.asyncio import Redis
from savant_rs.primitives import EndOfStream, VideoFrame, VideoFrameContent
from savant_rs.utils.serialization import Message, load_message_from_bytes

from adapters.python.shared.config import opt_config
from adapters.python.shared.kafka_redis import (
    STOP,
    BaseConfig,
    BaseKafkaConfig,
    BaseKafkaRedisAdapter,
    run_kafka_redis_adapter,
)
from savant.api.enums import ExternalFrameType
from savant.client import SourceBuilder
from savant.utils.logging import get_logger

LOGGER_NAME = 'adapters.kafka_redis_source'
logger = get_logger(LOGGER_NAME)


class KafkaConfig(BaseKafkaConfig):
    """Kafka configuration for kafka-redis source adapter."""

    def __init__(self):
        super().__init__()
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


class Config(BaseConfig):
    """Configuration for kafka-redis source adapter."""

    kafka: KafkaConfig

    def __init__(self):
        super().__init__()
        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.kafka = KafkaConfig()


class KafkaRedisSource(BaseKafkaRedisAdapter):
    """Kafka-redis source adapter."""

    _config: Config
    _poller_queue: Queue[bytes]
    _sender_queue: Queue[Union[Tuple[VideoFrame, bytes], EndOfStream]]

    def __init__(self, config: Config):
        super().__init__(config)
        self._source = (
            SourceBuilder()
            .with_socket(config.zmq_endpoint)
            .with_telemetry_disabled()
            .build_async()
        )
        self._consumer = build_consumer(config.kafka)
        self._frame_clients: Dict[str, Redis] = {}
        self._consumer.subscribe([self._config.kafka.topic])
        logger.info('Subscribed to topic %r', self._config.kafka.topic)

    async def poller(self):
        """Poll messages from Kafka topic and put them to the poller queue."""

        logger.info('Starting poller')
        loop = asyncio.get_running_loop()
        while self._is_running:
            logger.debug('Polling next message')
            try:
                message = await loop.run_in_executor(
                    None,
                    self._consumer.poll,
                    self._config.kafka.poll_timeout,
                )
            except Exception as e:
                self.set_error(f'Failed to poll message: {e}')
                break

            if message is None:
                continue

            if message.error() is not None:
                self.set_error(f'Failed to poll message: {message.error()}')
                break

            data = message.value()
            logger.debug(
                'Polled kafka message %s/%s/%s with key %s. Message size is %s bytes.',
                message.topic(),
                message.partition(),
                message.offset(),
                message.key(),
                len(data),
            )
            await self._poller_queue.put(data)

        await self._poller_queue.put(STOP)
        logger.info('Poller was stopped')

    async def messages_processor(self):
        """Process messages from the poller queue and put them to the sender queue.

        Frame content is fetched from Redis and frame metadata is updated.
        """

        logger.info('Starting deserializer')
        while self._error is None:
            logger.debug('Waiting for the next message')
            data = await self._poller_queue.get()
            if data is STOP:
                logger.debug('Received stop signal')
                self._poller_queue.task_done()
                break

            try:
                deserialized = await self.process_message(data)
            except Exception as e:
                self._is_running = False
                self.set_error(f'Failed to deserialize message: {e}')
                self._poller_queue.task_done()
                # In case self._poller_queue is full, so poller won't stuck
                self.clear_queue(self._poller_queue)
                break

            if deserialized is not None:
                await self._sender_queue.put(deserialized)
            self._poller_queue.task_done()

        await self._sender_queue.put(STOP)
        logger.info('Deserializer was stopped')

    async def sender(self):
        """Send messages from the sender queue to ZeroMQ socket."""

        logger.info('Starting sender')
        try:
            async for result in self._source.send_iter(self.video_frame_iterator()):
                logger.debug(
                    'Status of sending frame for source %s: %s.',
                    result.source_id,
                    result.status,
                )
        except Exception as e:
            self._is_running = False
            self.set_error(f'Failed to send frame: {e}')
            # In case self._sender_queue is full, so deserializer won't stuck
            self.clear_queue(self._sender_queue)
        logger.info('Sender was stopped')

    async def process_message(
        self,
        data: bytes,
    ) -> Optional[Union[Tuple[VideoFrame, bytes], EndOfStream]]:
        """Process one message from the poller queue.

        Frame content is fetched from Redis and frame metadata is updated.
        """

        message: Message = load_message_from_bytes(data)
        if message.is_video_frame():
            video_frame = message.as_video_frame()
            return await self.fetch_video_frame_content(video_frame)
        if message.is_end_of_stream():
            return message.as_end_of_stream()

        if message.is_unknown():
            raise RuntimeError(f'Unknown message: {message}')

    async def fetch_video_frame_content(
        self,
        video_frame: VideoFrame,
    ) -> Optional[Tuple[VideoFrame, bytes]]:
        """Fetch frame content from Redis and update frame metadata."""

        if video_frame.content.is_internal():
            content = video_frame.content.get_data_as_bytes()

        elif video_frame.content.is_external():
            if video_frame.content.get_method() != ExternalFrameType.REDIS.value:
                logger.warning(
                    'Unsupported external frame type %r',
                    video_frame.content.get_method(),
                )
                return None
            content = await self.fetch_content_from_redis(
                video_frame.content.get_location()
            )
            if content is None:
                logger.warning(
                    'Failed to fetch frame from %r',
                    video_frame.content.get_location(),
                )
                return None

        else:
            logger.warning('Unsupported frame content %r', video_frame.content)
            return None

        video_frame.content = VideoFrameContent.external(
            ExternalFrameType.ZEROMQ.value, None
        )
        return video_frame, content

    async def video_frame_iterator(
        self,
    ) -> AsyncIterator[Union[Tuple[VideoFrame, bytes], EndOfStream]]:
        """Iterate over messages from the sender queue."""

        while self._error is None:
            logger.debug('Waiting for the next message')
            data = await self._sender_queue.get()
            if data is STOP:
                logger.debug('Received stop signal')
                self._sender_queue.task_done()
                break
            yield data
            self._sender_queue.task_done()

    async def fetch_content_from_redis(self, location: str) -> Optional[bytes]:
        """Fetch frame content from Redis."""

        logger.debug('Fetching frame from %r', location)
        host_port, key = location.split('/', 1)
        frame_client = self._frame_clients.get(host_port)

        if frame_client is None:
            logger.info('Connecting to %r', host_port)
            host, port = host_port.split(':')
            frame_client = Redis(host=host, port=int(port))
            self._frame_clients[host_port] = frame_client

        return await frame_client.get(key)


def build_consumer(config: KafkaConfig) -> Consumer:
    """Build Kafka consumer."""

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


if __name__ == '__main__':
    run_kafka_redis_adapter(KafkaRedisSource, Config)
