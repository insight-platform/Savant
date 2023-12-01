import asyncio
import os
import time
from asyncio import Queue
from typing import Tuple

from confluent_kafka import KafkaError, Producer
from redis.asyncio import Redis
from savant_rs.primitives import (
    VideoFrame,
    VideoFrameContent,
    VideoFrameTranscodingMethod,
)
from savant_rs.utils.serialization import Message, save_message_to_bytes

from adapters.python.shared.config import opt_config
from adapters.python.shared.kafka_redis import (
    STOP,
    BaseConfig,
    BaseKafkaConfig,
    BaseKafkaRedisAdapter,
    run_kafka_redis_adapter,
)
from savant.api.enums import ExternalFrameType
from savant.client import SinkBuilder
from savant.client.runner.sink import SinkResult


class KafkaConfig(BaseKafkaConfig):
    """Kafka configuration for kafka-redis sink adapter."""

    def __init__(self):
        super().__init__()
        self.flush_interval = opt_config('KAFKA_FLUSH_INTERVAL', 1, float)
        self.flush_timeout = opt_config('KAFKA_FLUSH_TIMEOUT', 10, float)


class RedisConfig:
    """Redis configuration for kafka-redis sink adapter."""

    def __init__(self):
        self.host = os.environ['REDIS_HOST']
        self.port = opt_config('REDIS_PORT', 6379, int)
        self.db = opt_config('REDIS_DB', 0, int)
        self.key_prefix = opt_config('REDIS_KEY_PREFIX', 'savant:frames')
        self.ttl_seconds = opt_config('REDIS_TTL_SECONDS', 60, int)


class Config(BaseConfig):
    """Configuration for kafka-redis sink adapter."""

    kafka: KafkaConfig

    def __init__(self):
        super().__init__()
        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.deduplicate = opt_config('DEDUPLICATE', False, bool)
        self.kafka = KafkaConfig()
        try:
            self.redis = RedisConfig()
        except KeyError:
            self.redis = None


class KafkaRedisSink(BaseKafkaRedisAdapter):
    """Kafka-redis sink adapter."""

    _config: Config
    _poller_queue: Queue[SinkResult]
    _sender_queue: Queue[Tuple[bytes, bytes]]

    def __init__(self, config: Config):
        super().__init__(config)
        self._last_flush_ts = 0
        self._producer = self.build_producer()
        if config.redis is not None:
            self._logger.info(
                'Redis is configured at %s:%s/%s. Frame content will be stored to Redis.',
                config.redis.host,
                config.redis.port,
                config.redis.db,
            )
            self._redis_client = Redis(
                host=config.redis.host,
                port=config.redis.port,
                db=config.redis.db,
            )
            self._logger.info(
                'Deduplication is %s.',
                'enabled' if config.deduplicate else 'disabled',
            )
        else:
            self._logger.info(
                'Redis is not configured. Frame content will be stored internally.'
            )
            self._redis_client = None
        self._sink = (
            SinkBuilder()
            .with_socket(config.zmq_endpoint)
            .with_idle_timeout(3)  # For graceful shutdown
            .build_async()
        )

    async def on_start(self):
        pass

    async def flush_producer(self, loop: asyncio.AbstractEventLoop):
        self._logger.debug('Flushing pending producer messages')
        try:
            loop.run_in_executor(
                None,
                self._producer.flush,
                self._config.kafka.flush_timeout,
            )
            self._last_flush_ts = time.time()
        except Exception as e:
            self.set_error(f'Failed to flush producer: {e}')

    async def on_stop(self):
        await self.flush_producer(asyncio.get_running_loop())

    async def poller(self):
        """Poll messages from ZeroMQ socket and put them to the poller queue."""

        self._logger.info('Starting poller')
        loop = asyncio.get_running_loop()
        self._last_flush_ts = time.time()
        while self._is_running:
            try:
                async for result in self._sink:
                    if not self._is_running:
                        break
                    await self._poller_queue.put(result)
                    if (
                        time.time() - self._last_flush_ts
                        > self._config.kafka.flush_interval
                    ):
                        await self.flush_producer(loop)
            except Exception as e:
                self._is_running = False
                self.set_error(f'Failed to poll message: {e}')
                break
        await self._poller_queue.put(STOP)
        self._logger.info('Poller was stopped')

    async def messages_processor(self):
        """Process messages from the poller queue and put them to the sender queue.

        Frame content is saved to Redis and frame metadata is updated with the content location.
        """

        self._logger.info('Starting serializer')
        while self._error is None:
            result = await self._poller_queue.get()
            if result is STOP:
                self._logger.debug('Received stop signal')
                self._poller_queue.task_done()
                break
            try:
                message, source_id = await self.process_message(result)
            except Exception as e:
                self._is_running = False
                self.set_error(f'Failed to serialize message: {e}')
                self._poller_queue.task_done()
                # In case self._poller_queue is full, so poller won't stuck
                self.clear_queue(self._poller_queue)
                break
            await self._sender_queue.put(
                (source_id.encode(), save_message_to_bytes(message))
            )
            self._poller_queue.task_done()
        await self._sender_queue.put(STOP)
        self._logger.info('Serializer was stopped')

    async def sender(self):
        """Send messages from the sender queue to Kafka topic."""

        self._logger.info('Starting sender')
        loop = asyncio.get_running_loop()
        while self._error is None:
            message = await self._sender_queue.get()
            if message is STOP:
                self._logger.debug('Received stop signal')
                self._sender_queue.task_done()
                break
            source_id, data = message
            try:
                await loop.run_in_executor(None, self.send_to_producer, source_id, data)
            except Exception as e:
                self._is_running = False
                self.set_error(f'Failed to send message: {e}')
                self._sender_queue.task_done()
                # In case self._sender_queue is full, so serializer won't stuck
                self.clear_queue(self._sender_queue)
                break
            self._sender_queue.task_done()

        self._logger.info('Sender was stopped')

    async def process_message(self, result: SinkResult):
        """Process one message from the poller queue.

        Frame content is saved to Redis and frame metadata is updated with the content location.
        """

        if result.frame_meta is not None:
            frame_meta = result.frame_meta
            source_id = frame_meta.source_id
            self._logger.debug(
                'Received frame %s/%s (keyframe=%s)',
                source_id,
                frame_meta.pts,
                frame_meta.keyframe,
            )
            if result.frame_content is not None:
                frame_meta.content = await self.store_frame_content(
                    frame_meta, result.frame_content
                )
            message = Message.video_frame(frame_meta)
            self.count_frame()

        else:
            source_id = result.eos.source_id
            self._logger.debug('Received EOS for source %s', source_id)
            message = Message.end_of_stream(result.eos)

        return message, source_id

    async def store_frame_content(
        self,
        frame: VideoFrame,
        content: bytes,
    ) -> VideoFrameContent:
        """Store frame content.

        If Redis is configured, store frame content to Redis and update frame
        metadata with the content location. Otherwise, store frame content
        directly to the VideoFrame.
        """

        if self._redis_client is None:
            self._logger.debug(
                'Storing content of the frame %s from source %s internally (%s bytes)',
                frame.source_id,
                frame.pts,
                len(content),
            )
            return VideoFrameContent.internal(content)

        content_key = f'{self._config.redis.key_prefix}:{frame.uuid}'
        location = f'{self._config.redis.host}:{self._config.redis.port}:{self._config.redis.db}/{content_key}'

        if await self.need_to_store_content_to_redis(frame, content_key, location):
            self._logger.debug(
                'Storing content of the frame %s from source %s to Redis location %r (%s bytes)',
                frame.pts,
                frame.source_id,
                location,
                len(content),
            )
            await self._redis_client.set(
                content_key,
                content,
                ex=self._config.redis.ttl_seconds,
            )

        return VideoFrameContent.external(ExternalFrameType.REDIS.value, location)

    async def need_to_store_content_to_redis(
        self,
        frame: VideoFrame,
        content_key: str,
        location: str,
    ):
        """Check if the frame content needs to be stored to Redis.

        The frame content doesn't need to be stored in Redis when the following conditions are met:
        - the deduplication is enabled;
        - the module before the adapter works in pass-through mode;
        - Redis already contains the frame content at the location.

        In that case only TTL of the content in Redis is updated. Otherwise,
        the frame content should be stored in Redis.
        """

        if not self._config.deduplicate:
            return True

        if frame.transcoding_method == VideoFrameTranscodingMethod.Encoded:
            self._logger.debug(
                'Content of the frame %s from source %s was modified.',
                location,
                frame.source_id,
            )
            return True

        if await self._redis_client.expire(
            content_key,
            self._config.redis.ttl_seconds,
        ):
            self._logger.debug(
                'Content of the frame %s from source %s is already in Redis at %r. TTL was updated.',
                frame.pts,
                frame.source_id,
                location,
            )
            return False

        self._logger.debug(
            'Content of the frame %s from source %s is not in Redis at %r.',
            frame.pts,
            frame.source_id,
            location,
        )
        return True

    def send_to_producer(self, key: str, value: bytes):
        """Send message to Kafka topic."""

        self._logger.debug(
            'Sending message to kafka topic %s with key %s. Message size is %s bytes.',
            self._config.kafka.topic,
            key,
            len(value),
        )
        self._logger.trace('Value: %s.', value)
        self._producer.produce(self._config.kafka.topic, key=key, value=value)

    def on_producer_error(self, error: KafkaError):
        """Handle producer error."""

        self.set_error(f'Failed to produce message: {error}')
        self._is_running = False

    def build_producer(self) -> Producer:
        """Build Kafka producer."""

        return Producer(
            {
                'bootstrap.servers': self._config.kafka.brokers,
                'error_cb': self.on_producer_error,
            }
        )


if __name__ == '__main__':
    run_kafka_redis_adapter(KafkaRedisSink, Config)
