import asyncio
import json
import os
import signal
import time
from abc import ABC, abstractmethod
from asyncio import Event, Queue
from distutils.util import strtobool
from typing import Any, Dict, Optional, Type

from confluent_kafka.admin import AdminClient, ClusterMetadata, NewTopic

from savant.utils.logging import get_logger, init_logging

from .config import opt_config

logger = get_logger(__name__)


class BaseKafkaConfig:
    def __init__(self):
        self.brokers = os.environ['KAFKA_BROKERS']
        self.topic = os.environ['KAFKA_TOPIC']
        self.create_topic = opt_config('KAFKA_CREATE_TOPIC', False, strtobool)
        self.create_topic_config: Dict[str, Any] = opt_config(
            'KAFKA_CREATE_TOPIC_CONFIG',
            {},
            json.loads,
        )


class BaseConfig:
    kafka: BaseKafkaConfig

    def __init__(self):
        self.queue_size = opt_config('QUEUE_SIZE', 100, int)


STOP = object()


class BaseKafkaRedisAdapter(ABC):
    _config: BaseConfig
    _poller_queue: Queue
    _sender_queue: Queue
    _stop_event: Event

    def __init__(self, config: BaseConfig):
        self._config = config
        self._is_running = False
        self._error: Optional[str] = None

    async def run(self):
        logger.info('Starting adapter')
        if not self.kafka_topic_exists():
            raise RuntimeError(
                f'Topic {self._config.kafka.topic} does not exist and '
                f'KAFKA_CREATE_TOPIC={self._config.kafka.create_topic}'
            )
        self._poller_queue = Queue(self._config.queue_size)
        self._sender_queue = Queue(self._config.queue_size)
        self._stop_event = Event()
        await self.prepare_to_run()
        self._is_running = True
        await asyncio.gather(self.poller(), self.transformer(), self.sender())
        self._stop_event.set()

    @abstractmethod
    async def prepare_to_run(self):
        pass

    async def stop(self):
        logger.info('Stopping adapter')
        self._is_running = False
        await self._stop_event.wait()
        logger.info('Adapter was stopped')

    @abstractmethod
    async def poller(self):
        pass

    @abstractmethod
    async def transformer(self):
        pass

    @abstractmethod
    async def sender(self):
        pass

    @property
    def error(self) -> Optional[str]:
        return self._error

    def set_error(self, error: str):
        logger.error(error)
        if self._error is None:
            self._error = error

    def kafka_topic_exists(self, timeout: int = 10):
        admin_client = AdminClient({'bootstrap.servers': self._config.kafka.brokers})
        cluster_meta: ClusterMetadata = admin_client.list_topics()
        if self._config.kafka.topic in cluster_meta.topics:
            return True

        if not self._config.kafka.create_topic:
            raise False

        logger.info(
            'Creating kafka topic %s with config %s',
            self._config.kafka.topic,
            self._config.kafka.create_topic_config,
        )
        admin_client.create_topics(
            [
                NewTopic(
                    self._config.kafka.topic,
                    **self._config.kafka.create_topic_config,
                )
            ]
        )
        for _ in range(timeout):
            cluster_meta = admin_client.list_topics()
            if self._config.kafka.topic in cluster_meta.topics:
                return True
            time.sleep(1)

        logger.error('Failed to create kafka topic %s', self._config.kafka.topic)

        return False

    def clear_queue(self, queue: Queue):
        while not queue.empty():
            queue.get_nowait()


def run_kafka_redis_adapter(
    adapter_class: Type[BaseKafkaRedisAdapter],
    config_class: Type[BaseConfig],
):
    init_logging()
    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))
    source = adapter_class(config_class())
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(source.run())
    except KeyboardInterrupt:
        loop.run_until_complete(source.stop())
    finally:
        loop.close()
    if source.error is not None:
        logger.error(source.error)
        exit(1)
