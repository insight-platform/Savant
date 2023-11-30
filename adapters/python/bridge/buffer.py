import json
import os
import signal
import time
from typing import Dict, List, Optional

import zmq
from rocksq.blocking import PersistentQueueWithCapacity
from savant_rs.pipeline2 import (
    VideoPipeline,
    VideoPipelineConfiguration,
    VideoPipelineStagePayloadType,
)
from savant_rs.primitives import VideoFrame, VideoFrameContent
from savant_rs.utils.serialization import Message, load_message_from_bytes

from adapters.python.shared.config import opt_config
from adapters.shared.thread import BaseThreadWorker
from savant.metrics import Counter, Gauge
from savant.metrics.prometheus import BaseMetricsCollector, PrometheusMetricsExporter
from savant.utils.logging import get_logger, init_logging
from savant.utils.zeromq import (
    Defaults,
    SenderSocketTypes,
    ZeroMQSource,
    parse_zmq_socket_uri,
)

LOGGER_NAME = 'adapters.buffer'
# For each message we need 3 slots: source ID, metadata, frame content
QUEUE_ITEM_SIZE = 3
logger = get_logger(LOGGER_NAME)


class MetricsConfig:
    """Metrics configuration for the adapter."""

    def __init__(self):
        self.frame_period = opt_config('METRICS_FRAME_PERIOD', 1000, int)
        self.time_period = opt_config('METRICS_TIME_PERIOD', convert=float)
        self.history = opt_config('METRICS_HISTORY', 100, int)
        self.provider = opt_config('METRICS_PROVIDER')
        self.provider_params: dict = opt_config(
            'METRICS_PROVIDER_PARAMS', {}, json.loads
        )


class BufferConfig:
    """Buffer configuration for the adapter."""

    def __init__(self):
        self.path = os.environ['BUFFER_PATH']
        len_items = opt_config('BUFFER_LEN', 1000, int)
        assert len_items > 0, 'BUFFER_LEN must be positive'
        self.len = len_items * QUEUE_ITEM_SIZE

        service_messages_items = opt_config('BUFFER_SERVICE_MESSAGES', 100, int)
        assert service_messages_items > 0, 'BUFFER_SERVICE_MESSAGES must be positive'
        self.service_messages = service_messages_items * QUEUE_ITEM_SIZE

        threshold_percentage = opt_config('BUFFER_THRESHOLD_PERCENTAGE', 80, int)
        assert (
            0 <= threshold_percentage <= 100
        ), 'BUFFER_THRESHOLD_PERCENTAGE must be in [0, 100] range'
        self.threshold = int(len_items * threshold_percentage / 100) * QUEUE_ITEM_SIZE


class Config:
    """Configuration for the adapter."""

    def __init__(self):
        self.zmq_src_endpoint = os.environ['ZMQ_SRC_ENDPOINT']
        self.zmq_sink_endpoint = os.environ['ZMQ_SINK_ENDPOINT']
        self.buffer = BufferConfig()
        self.idle_polling_period = opt_config('IDLE_POLLING_PERIOD', 0.005, float)
        self.stats_log_interval = opt_config('STATS_LOG_INTERVAL', 60, int)
        self.metrics = MetricsConfig()


class Ingress(BaseThreadWorker):
    """Receives messages from the source ZeroMQ socket and pushes them to the buffer."""

    def __init__(self, queue: PersistentQueueWithCapacity, config: Config):
        super().__init__(
            thread_name='Ingress',
            logger_name=f'{LOGGER_NAME}.{self.__class__.__name__}',
            daemon=True,
        )
        self._queue = queue
        self._config = config
        self._buffer_is_full = False
        self._received_messages = 0
        self._pushed_messages = 0
        self._dropped_messages = 0
        self._zmq_source = ZeroMQSource(config.zmq_src_endpoint)

    def workload(self):
        self.logger.info('Starting Ingress')
        self._zmq_source.start()
        while self.is_running:
            try:
                message_parts = self._zmq_source.next_message_without_routing_id()
                if message_parts is not None:
                    self.logger.debug('Received message from the source ZeroMQ socket')
                    self.handle_next_message(message_parts)
            except Exception as e:
                self.logger.error('Failed to poll message: %s', e)
                self.is_running = False
                break
        self.logger.info('Ingress was stopped')

    def handle_next_message(self, message_parts: List[bytes]):
        """Handle the next message from the source ZeroMQ socket."""

        self._received_messages += 1
        while len(message_parts) < QUEUE_ITEM_SIZE:
            message_parts.append(b'')
        message: Message = load_message_from_bytes(message_parts[1])
        if message.is_video_frame():
            pushed = self.push_frame(message_parts)
        else:
            pushed = self.push_service_message(message_parts)
        if pushed:
            self._pushed_messages += 1
        else:
            self._dropped_messages += 1

    def push_frame(self, message_parts: List[bytes]) -> bool:
        """Push frame to the buffer."""

        buffer_size = self._queue.len()
        if self._buffer_is_full:
            if buffer_size >= self._config.buffer.threshold:
                self.logger.debug('Buffer is full, dropping the frame')
                return False

            self._buffer_is_full = False

        elif buffer_size + QUEUE_ITEM_SIZE >= self._config.buffer.len:
            self._buffer_is_full = True
            self.logger.debug('Buffer is full, dropping the frame')
            return False

        self._queue.push(message_parts)
        self.logger.debug('Pushed frame to the buffer')
        return True

    def push_service_message(self, message_parts: List[bytes]) -> bool:
        """Push service message to the buffer."""

        try:
            self._queue.push(message_parts)
        except Exception as e:
            if e.args[0] != 'Failed to push item: Queue is full':
                raise
            self.logger.debug('Buffer is full, dropping the message')
            return False

        self.logger.debug('Pushed message to the buffer')
        return True

    @property
    def received_messages(self) -> int:
        """Number of messages received from the source ZeroMQ socket."""
        return self._received_messages

    @property
    def pushed_messages(self) -> int:
        """Number of messages pushed to the buffer."""
        return self._pushed_messages

    @property
    def dropped_messages(self) -> int:
        """Number of messages dropped by the adapter."""
        return self._dropped_messages


class Egress(BaseThreadWorker):
    """Polls messages from the buffer and sends them to the sink ZeroMQ socket."""

    def __init__(
        self,
        queue: PersistentQueueWithCapacity,
        pipeline: VideoPipeline,
        config: Config,
    ):
        super().__init__(
            thread_name='Egress',
            logger_name=f'{LOGGER_NAME}.{self.__class__.__name__}',
            daemon=True,
        )
        self._queue = queue
        self._idle_polling_period = config.idle_polling_period
        self._sent_messages = 0
        self._pipeline = pipeline
        self._video_frame = VideoFrame(
            source_id='test',
            framerate='30/1',
            width=1280,
            height=720,
            content=VideoFrameContent.none(),
        )

        self._socket_type, self._bind, self._socket = parse_zmq_socket_uri(
            uri=config.zmq_sink_endpoint,
            socket_type_enum=SenderSocketTypes,
            socket_type_name=None,
            bind=None,
        )
        assert (
            self._socket_type == SenderSocketTypes.DEALER
        ), 'Only DEALER socket type is supported for Egress'
        self._zmq_context = zmq.Context()
        self._sender: zmq.Socket = self._zmq_context.socket(self._socket_type.value)
        self._sender.setsockopt(zmq.SNDHWM, Defaults.SEND_HWM)
        self._sender.setsockopt(zmq.RCVTIMEO, Defaults.SENDER_RECEIVE_TIMEOUT)
        if self._bind:
            self._sender.bind(self._socket)
        else:
            self._sender.connect(self._socket)

    def workload(self):
        self.logger.info('Starting Egress')
        while self.is_running:
            try:
                message_parts = self.pop_next_message()
                if message_parts is not None:
                    self.logger.debug('Sending message to the sink ZeroMQ socket')
                    self._sender.send_multipart(message_parts)
            except Exception as e:
                self.logger.error('Failed to poll message: %s', e)
                self.is_running = False
                break
        self.logger.info('Egress was stopped')

    def pop_next_message(self) -> Optional[List[bytes]]:
        """Pop the next message from the buffer.

        When the buffer is empty, wait until it is not empty and then pop the message.
        """

        if self._queue.len() < QUEUE_ITEM_SIZE:
            self.logger.trace(
                'Buffer is empty, waiting for %s seconds',
                self._idle_polling_period,
            )
            time.sleep(self._idle_polling_period)
            return None

        message_parts = self._queue.pop(QUEUE_ITEM_SIZE)
        if not message_parts[-1]:
            message_parts.pop()
        self._sent_messages += 1
        frame_id = self._pipeline.add_frame('fps-meter', self._video_frame)
        self._pipeline.delete(frame_id)

        return message_parts

    @property
    def sent_messages(self) -> int:
        """Number of messages sent to the sink ZeroMQ socket."""
        return self._sent_messages


class StatsAggregator:
    """Aggregates statistics from the adapter threads."""

    def __init__(
        self,
        queue: PersistentQueueWithCapacity,
        ingress: Ingress,
        egress: Egress,
    ):
        self._queue = queue
        self._ingress = ingress
        self._egress = egress

    def get_stats(self):
        """Get statistics from the adapter threads.

        The following statistics are returned:
        - received_messages: number of messages received from the source ZeroMQ socket;
        - pushed_messages: number of messages pushed to the buffer;
        - dropped_messages: number of messages dropped by the adapter;
        - sent_messages: number of messages sent to the sink ZeroMQ socket;
        - buffer_size: number of messages in the buffer.
        """

        received_messages = self._ingress.received_messages
        pushed_messages = self._ingress.pushed_messages
        dropped_messages = self._ingress.dropped_messages
        sent_messages = self._egress.sent_messages
        buffer_size = self._queue.len() // QUEUE_ITEM_SIZE

        return {
            'received_messages': received_messages,
            'pushed_messages': pushed_messages,
            'dropped_messages': dropped_messages,
            'sent_messages': sent_messages,
            'buffer_size': buffer_size,
        }


class StatsLogger(BaseThreadWorker):
    """Logs stats."""

    def __init__(
        self,
        stats_aggregator: StatsAggregator,
        config: Config,
    ):
        super().__init__(
            thread_name='StatsLogger',
            logger_name=f'{LOGGER_NAME}.{self.__class__.__name__}',
            daemon=True,
        )
        self._stats_aggregator = stats_aggregator
        self._interval = config.stats_log_interval

    def workload(self):
        self.logger.info('Starting StatsLogger')
        while self.is_running:
            time.sleep(self._interval)
            self.log_stats()
        self.logger.info('StatsLogger was stopped')

    def log_stats(self):
        """Log stats from the adapter threads."""

        stats = self._stats_aggregator.get_stats()
        self.logger.info(
            'Received %s, pushed %s, dropped %s, sent %s, buffer size %s',
            stats['received_messages'],
            stats['pushed_messages'],
            stats['dropped_messages'],
            stats['sent_messages'],
            stats['buffer_size'],
        )


class AdapterMetricsCollector(BaseMetricsCollector):
    """Adapter metrics collector for Prometheus."""

    def __init__(
        self,
        metrics_aggregator: StatsAggregator,
        extra_labels: Dict[str, str],
    ):
        super().__init__(extra_labels)
        self._metrics_aggregator = metrics_aggregator
        self.register_metric(
            Counter('received_messages', 'Number of messages received by the adapter')
        )
        self.register_metric(
            Counter('pushed_messages', 'Number of messages pushed to the buffer')
        )
        self.register_metric(
            Counter('dropped_messages', 'Number of messages dropped by the buffer')
        )
        self.register_metric(
            Counter('sent_messages', 'Number of messages sent by the adapter')
        )
        self.register_metric(Gauge('buffer_size', 'Number of messages in the buffer'))

    def update_all_metrics(self):
        for k, v in self._metrics_aggregator.get_stats().items():
            self._metrics[k].set(v)


def build_video_pipeline(config: Config):
    """Build a video pipeline to count passed frames."""

    conf = VideoPipelineConfiguration()
    conf.frame_period = (
        config.metrics.frame_period if config.metrics.frame_period else None
    )
    conf.timestamp_period = (
        int(config.metrics.time_period * 1000) if config.metrics.time_period else None
    )
    conf.collection_history = config.metrics.history

    return VideoPipeline(
        'buffer-adapter',
        [('fps-meter', VideoPipelineStagePayloadType.Frame)],
        conf,
    )


def main():
    init_logging()
    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))
    config = Config()
    logger.info('Starting the adapter')
    queue = PersistentQueueWithCapacity(
        config.buffer.path,
        config.buffer.len + config.buffer.service_messages,
    )
    # VideoPipeline is used to count passed frames
    pipeline = build_video_pipeline(config)
    ingress = Ingress(queue, config)
    egress = Egress(queue, pipeline, config)
    stats_aggregator = StatsAggregator(queue, ingress, egress)
    stats_logger = StatsLogger(stats_aggregator, config)
    if config.metrics.provider is None:
        metrics_exporter = None
    elif config.metrics.provider == 'prometheus':
        metrics_exporter = PrometheusMetricsExporter(
            config.metrics.provider_params,
            AdapterMetricsCollector(
                stats_aggregator,
                config.metrics.provider_params.get('labels') or {},
            ),
        )
    else:
        raise ValueError(f'Unsupported metrics provider: {config.metrics.provider}')

    if metrics_exporter is not None:
        metrics_exporter.start()
    threads = [ingress, egress, stats_logger]
    for thread in threads:
        thread.start()
    try:
        while all(thread.is_running for thread in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info('Stopping the adapter')
    for thread in threads:
        thread.stop()
    for thread in threads:
        thread.join(3)
    stats_logger.log_stats()
    if metrics_exporter is not None:
        metrics_exporter.stop()
    for thread in threads:
        if thread.error is not None:
            logger.error(thread.error)
            exit(1)


if __name__ == '__main__':
    main()
