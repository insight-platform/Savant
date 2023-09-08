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

from savant.utils.fps_meter import FPSMeter
from savant.utils.logging import get_logger, init_logging

from .config import opt_config

logger = get_logger(__name__)


class FpsMeterConfig:
    """FPS measurement configuration."""

    def __init__(self):
        self.output = opt_config('FPS_OUTPUT', 'stdout', str)
        assert self.output in [
            'stdout',
            'logger',
        ], 'FPS_OUTPUT should be either "stdout" or "logger"'
        self.period_frames = opt_config('FPS_PERIOD_FRAMES', None, int)
        self.period_seconds = opt_config('FPS_PERIOD_SECONDS', None, float)
        if self.period_frames is None and self.period_seconds is None:
            self.period_frames = 1000


class BaseKafkaConfig:
    """Base class for kafka configuration."""

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
    """Base class for configuration."""

    kafka: BaseKafkaConfig

    def __init__(self):
        self.queue_size = opt_config('QUEUE_SIZE', 50, int)
        self.fps = FpsMeterConfig()


STOP = object()
"""Stop signal for the queues. Needed to gracefully stop the adapter."""


class BaseKafkaRedisAdapter(ABC):
    """Base class for kafka-redis adapters.

    The adapter works with asyncio.
    """

    _poller_queue: Queue
    _sender_queue: Queue
    _stop_event: Event

    def __init__(self, config: BaseConfig):
        self._config = config
        self._fps_meter = FPSMeter(
            period_frames=config.fps.period_frames,
            period_seconds=config.fps.period_seconds,
        )
        self._is_running = False
        self._error: Optional[str] = None

    async def run(self):
        """Run the adapter."""

        logger.info('Starting adapter')
        if not self.kafka_topic_exists():
            raise RuntimeError(
                f'Topic {self._config.kafka.topic} does not exist and '
                f'KAFKA_CREATE_TOPIC={self._config.kafka.create_topic}'
            )
        self._poller_queue = Queue(self._config.queue_size)
        self._sender_queue = Queue(self._config.queue_size)
        self._stop_event = Event()
        self._is_running = True
        self._fps_meter.start()
        await asyncio.gather(self.poller(), self.messages_processor(), self.sender())
        self.log_fps()
        self._stop_event.set()

    async def stop(self):
        """Gracefully stop the adapter."""

        logger.info('Stopping adapter')
        self._is_running = False
        await self._stop_event.wait()
        logger.info('Adapter was stopped')

    @abstractmethod
    async def poller(self):
        """Poll messages from the source and put them into the poller queue."""
        pass

    @abstractmethod
    async def messages_processor(self):
        """Process messages from the poller queue and put them into the sender queue."""
        pass

    @abstractmethod
    async def sender(self):
        """Send messages from the sender queue to the sink."""
        pass

    @property
    def error(self) -> Optional[str]:
        """Get the error message if the adapter failed to run."""
        return self._error

    def set_error(self, error: str):
        """Log and set the error message if the adapter failed to run.

        Only sets the first error.
        """

        logger.error(error, exc_info=True)
        if self._error is None:
            self._error = error

    def kafka_topic_exists(self, timeout: int = 10) -> bool:
        """Check if the kafka topic exists and create it if necessary.

        :param timeout: The timeout in seconds to wait for the topic to be created.
        """

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
        """Clear the queue. Needed to prevent the adapter from hanging in the case of failure."""

        while not queue.empty():
            queue.get_nowait()

    def count_frame(self):
        """Count frame for FPS measurement."""

        if self._fps_meter():
            self.log_fps()

    def log_fps(self):
        """Log FPS."""

        if self._config.fps.output == 'stdout':
            print(self._fps_meter.message)
        elif self._config.fps.output == 'logger':
            logger.info(self._fps_meter.message)


def run_kafka_redis_adapter(
    adapter_class: Type[BaseKafkaRedisAdapter],
    config_class: Type[BaseConfig],
):
    """Run kafka-redis adapter.

    :param adapter_class: The adapter implementation.
    :param config_class: The configuration implementation.
    """

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
