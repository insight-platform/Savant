import asyncio
import os
import uuid
from asyncio import Queue
from typing import Tuple

from confluent_kafka import Producer
from redis.asyncio import Redis
from savant_rs.primitives import VideoFrame, VideoFrameContent
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
from savant.utils.logging import get_logger

LOGGER_NAME = 'adapters.kafka_redis_sink'
logger = get_logger(LOGGER_NAME)


class KafkaConfig(BaseKafkaConfig):
    pass


class RedisConfig:
    def __init__(self):
        self.host = os.environ['REDIS_HOST']
        self.port = opt_config('REDIS_PORT', 6379, int)
        self.key_prefix = opt_config('REDIS_KEY_PREFIX', 'savant:frames')
        self.ttl_seconds = opt_config('REDIS_TTL_SECONDS', 60, int)

    @property
    def enabled(self):
        return self.host is not None and self.port is not None


class Config(BaseConfig):
    kafka: KafkaConfig

    def __init__(self):
        super().__init__()
        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.kafka = KafkaConfig()
        self.redis = RedisConfig()


class KafkaRedisSink(BaseKafkaRedisAdapter):
    _config: Config
    _poller_queue: Queue[SinkResult]
    _sender_queue: Queue[Tuple[bytes, bytes]]

    def __init__(self, config: Config):
        super().__init__(config)
        self._producer = build_producer(config.kafka)
        self._redis_client = Redis(host=config.redis.host, port=config.redis.port)
        self._sink = SinkBuilder().with_socket(config.zmq_endpoint).build_async()

    async def prepare_to_run(self):
        pass

    async def poller(self):
        logger.info('Starting poller')
        while self._is_running:
            try:
                async for result in self._sink:
                    if not self._is_running:
                        break
                    await self._poller_queue.put(result)
            except Exception as e:
                self._is_running = False
                self.set_error(f'Failed to poll message: {e}')
                break
        await self._poller_queue.put(STOP)
        logger.info('Poller was stopped')

    async def transformer(self):
        logger.info('Starting serializer')
        while self._error is None:
            result = await self._poller_queue.get()
            if result is STOP:
                logger.debug('Received stop signal')
                self._poller_queue.task_done()
                break
            try:
                message, source_id = await self.prepare_message(result)
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
        logger.info('Serializer was stopped')

    async def sender(self):
        logger.info('Starting sender')
        loop = asyncio.get_running_loop()
        while self._error is None:
            message = await self._sender_queue.get()
            if message is STOP:
                logger.debug('Received stop signal')
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
        logger.info('Sender was stopped')

    async def prepare_message(self, result: SinkResult):
        if result.frame_meta is not None:
            frame_meta = result.frame_meta
            source_id = frame_meta.source_id
            logger.debug(
                'Received frame %s/%s (keyframe=%s)',
                source_id,
                frame_meta.pts,
                frame_meta.keyframe,
            )
            if result.frame_content is not None:
                frame_meta = await self.put_frame_to_redis(
                    frame_meta, result.frame_content
                )
            message = Message.video_frame(frame_meta)
        else:
            source_id = result.eos.source_id
            logger.debug('Received EOS for source %s', source_id)
            message = Message.end_of_stream(result.eos)

        return message, source_id

    async def put_frame_to_redis(self, frame: VideoFrame, content: bytes):
        content_key = f'{self._config.redis.key_prefix}:{uuid.uuid4()}'
        await self._redis_client.set(
            content_key,
            content,
            ex=self._config.redis.ttl_seconds,
        )
        frame.content = VideoFrameContent.external(
            ExternalFrameType.REDIS.value,
            f'{self._config.redis.host}:{self._config.redis.port}/{content_key}',
        )
        return frame

    def send_to_producer(self, key: str, value: bytes):
        logger.debug(
            'Sending message to kafka topic %s with key %s. Message size is %s bytes. Value: %s.',
            self._config.kafka.topic,
            key,
            len(value),
            value,
        )
        self._producer.produce(self._config.kafka.topic, key=key, value=value)


def build_producer(config: KafkaConfig) -> Producer:
    return Producer({'bootstrap.servers': config.brokers})


if __name__ == '__main__':
    run_kafka_redis_adapter(KafkaRedisSink, Config)
