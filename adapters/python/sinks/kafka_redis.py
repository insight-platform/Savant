import asyncio
import json
import os
import signal
import uuid
from asyncio import Event, Queue
from distutils.util import strtobool
from typing import Any, Dict, Optional, Tuple, Union

from confluent_kafka import Producer
from redis.asyncio import Redis
from savant_rs.primitives import VideoFrame, VideoFrameContent
from savant_rs.utils.serialization import Message, save_message_to_bytes

from adapters.python.shared import kafka_topic_exists, opt_config
from savant.api.enums import ExternalFrameType
from savant.client import SinkBuilder
from savant.client.runner.sink import SinkResult
from savant.utils.logging import get_logger, init_logging

LOGGER_NAME = 'adapters.kafka_redis_sink'
logger = get_logger(LOGGER_NAME)

STOP = object()


class Config:
    def __init__(self):
        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.queue_size = opt_config('QUEUE_SIZE', 100, int)
        self.kafka = KafkaConfig()
        self.redis = RedisConfig()


class KafkaConfig:
    def __init__(self):
        self.brokers = os.environ['KAFKA_BROKERS']
        self.topic = os.environ['KAFKA_TOPIC']
        self.create_topic = opt_config('KAFKA_CREATE_TOPIC', False, strtobool)
        self.create_topic_config: Dict[str, Any] = opt_config(
            'KAFKA_CREATE_TOPIC_CONFIG',
            {},
            json.loads,
        )


class RedisConfig:
    def __init__(self):
        self.host = os.environ['REDIS_HOST']
        self.port = opt_config('REDIS_PORT', 6379, int)
        self.key_prefix = opt_config('REDIS_KEY_PREFIX', 'savant:frames')
        self.ttl_seconds = opt_config('REDIS_TTL_SECONDS', 60, int)

    @property
    def enabled(self):
        return self.host is not None and self.port is not None


class KafkaRedisSink:
    def __init__(self, config: Config):
        self._config = config
        self._producer = build_producer(config.kafka)
        self._redis_client = Redis(host=config.redis.host, port=config.redis.port)
        self._sink = SinkBuilder().with_socket(config.zmq_endpoint).build_async()
        self._is_running = False
        self._poller_queue: Optional[Queue[Union[SinkResult, STOP]]] = None
        self._sender_queue: Optional[Queue[Union[Tuple[bytes, bytes], STOP]]] = None
        self._stop_source_event: Optional[Event] = None

    async def run(self):
        logger.info('Starting the sink')
        if not kafka_topic_exists(
            self._config.kafka.brokers,
            self._config.kafka.topic,
            self._config.kafka.create_topic,
            self._config.kafka.create_topic_config,
        ):
            raise RuntimeError(
                f'Topic {self._config.kafka.topic} does not exist and '
                f'KAFKA_CREATE_TOPIC={self._config.kafka.create_topic}'
            )
        self._poller_queue = Queue(self._config.queue_size)
        self._sender_queue = Queue(self._config.queue_size)
        self._stop_source_event = Event()
        self._is_running = True
        await asyncio.gather(self.poller(), self.serializer(), self.sender())
        self._stop_source_event.set()

    async def poller(self):
        logger.info('Starting poller')
        while self._is_running:
            async for result in self._sink:
                await self._poller_queue.put(result)
        await self._poller_queue.put(STOP)
        logger.info('Poller was stopped')

    async def stop(self):
        logger.info('Stopping the sink')
        self._is_running = False
        await self._stop_source_event.wait()
        logger.info('Sink was stopped')

    async def serializer(self):
        logger.info('Starting serializer')
        while True:
            result = await self._poller_queue.get()
            if result is STOP:
                logger.debug('Received stop signal')
                self._poller_queue.task_done()
                break
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
            await self._sender_queue.put(
                (source_id.encode(), save_message_to_bytes(message))
            )
            self._poller_queue.task_done()
        await self._sender_queue.put(STOP)
        logger.info('Serializer was stopped')

    async def sender(self):
        logger.info('Starting sender')
        loop = asyncio.get_running_loop()
        while True:
            message = await self._sender_queue.get()
            if message is STOP:
                logger.debug('Received stop signal')
                self._sender_queue.task_done()
                break
            source_id, data = message
            await loop.run_in_executor(None, self.send_to_producer, source_id, data)
            self._sender_queue.task_done()
        logger.info('Sender was stopped')

    def send_to_producer(self, key: str, value: bytes):
        logger.debug(
            'Sending message to kafka topic %s with key %s. Message size is %s bytes.',
            self._config.kafka.topic,
            key,
            len(value),
        )
        self._producer.produce(self._config.kafka.topic, key=key, value=value)

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


def build_producer(config: KafkaConfig) -> Producer:
    return Producer({'bootstrap.servers': config.brokers})


def main():
    init_logging()
    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))
    source = KafkaRedisSink(Config())
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(source.run())
    except KeyboardInterrupt:
        loop.run_until_complete(source.stop())
    finally:
        loop.close()


if __name__ == '__main__':
    main()
