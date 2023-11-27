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
    def __init__(self):
        self.frame_period = opt_config('METRICS_FRAME_PERIOD', 1000, int)
        self.time_period = opt_config('METRICS_TIME_PERIOD', convert=float)
        self.history = opt_config('METRICS_HISTORY', 100, int)
        self.provider = opt_config('METRICS_PROVIDER')
        self.provider_params: dict = opt_config(
            'METRICS_PROVIDER_PARAMS', {}, json.loads
        )


class Config:
    def __init__(self):
        self.zmq_src_endpoint = os.environ['ZMQ_SRC_ENDPOINT']
        self.zmq_sink_endpoint = os.environ['ZMQ_SINK_ENDPOINT']
        self.queue_path = os.environ['QUEUE_PATH']
        self.queue_capacity = opt_config('QUEUE_CAPACITY', 1000, int)
        self.interval = opt_config('INTERVAL', 1, float)
        self.stats_log_interval = opt_config('STATS_LOG_INTERVAL', 60, int)
        self.metrics = MetricsConfig()


class Ingress(BaseThreadWorker):
    def __init__(self, queue: PersistentQueueWithCapacity, config: Config):
        super().__init__(
            thread_name='Ingress',
            logger_name=f'{LOGGER_NAME}.{self.__class__.__name__}',
            daemon=True,
        )
        self._queue = queue
        self._interval = config.interval
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
                    self.handle_next_message(message_parts)
            except Exception as e:
                self.logger.error('Failed to poll message: %s', e)
                self.is_running = False
                break
        self.logger.info('Ingress was stopped')

    def handle_next_message(self, message_parts: List[bytes]):
        self._received_messages += 1
        while len(message_parts) < QUEUE_ITEM_SIZE:
            message_parts.append(b'')
        message: Message = load_message_from_bytes(message_parts[1])
        if message.is_video_frame():
            pushed = self.push(message_parts)
        else:
            pushed = self.force_push(message_parts)
        if pushed:
            self._pushed_messages += 1
        else:
            self._dropped_messages += 1

    def push(self, message_parts: List[bytes]) -> bool:
        try:
            self._queue.push(message_parts)
        except RuntimeError:
            self.logger.debug('Queue is full, dropping the message')
            return False
        self.logger.debug('Pushed message to the queue')
        return True

    def force_push(self, message_parts: List[bytes]) -> bool:
        while True:
            try:
                self._queue.push(message_parts)
                break
            except RuntimeError:
                self.logger.debug(
                    'Queue is full, retrying in %s seconds', self._interval
                )
                time.sleep(self._interval)

        self.logger.debug('Pushed message to the queue')
        return True

    @property
    def received_messages(self) -> int:
        return self._received_messages

    @property
    def pushed_messages(self) -> int:
        return self._pushed_messages

    @property
    def dropped_messages(self) -> int:
        return self._dropped_messages


class Egress(BaseThreadWorker):
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
        self._interval = config.interval
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
                    self._sender.send_multipart(message_parts)
            except Exception as e:
                self.logger.error('Failed to poll message: %s', e)
                self.is_running = False
                break
        self.logger.info('Egress was stopped')

    def pop_next_message(self) -> Optional[List[bytes]]:
        if self._queue.len() < QUEUE_ITEM_SIZE:
            self.logger.debug('Queue is empty, waiting for %s seconds', self._interval)
            time.sleep(self._interval)
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
        return self._sent_messages


class MetricsAggregator:
    def __init__(
        self, queue: PersistentQueueWithCapacity, ingress: Ingress, egress: Egress
    ):
        self._queue = queue
        self._ingress = ingress
        self._egress = egress

    def get_metrics(self):
        received_messages = self._ingress.received_messages
        pushed_messages = self._ingress.pushed_messages
        dropped_messages = self._ingress.dropped_messages
        sent_messages = self._egress.sent_messages
        queue_size = self._queue.len() // QUEUE_ITEM_SIZE

        return {
            'received_messages': received_messages,
            'pushed_messages': pushed_messages,
            'dropped_messages': dropped_messages,
            'sent_messages': sent_messages,
            'queue_size': queue_size,
        }


class StatsLogger(BaseThreadWorker):
    def __init__(
        self,
        metrics_aggregator: MetricsAggregator,
        config: Config,
    ):
        super().__init__(
            thread_name='StatsLogger',
            logger_name=f'{LOGGER_NAME}.{self.__class__.__name__}',
            daemon=True,
        )
        self._metrics_aggregator = metrics_aggregator
        self._interval = config.stats_log_interval

    def workload(self):
        self.logger.info('Starting StatsLogger')
        while self.is_running:
            time.sleep(self._interval)
            self.log_stats()
        self.logger.info('StatsLogger was stopped')

    def log_stats(self):
        metrics = self._metrics_aggregator.get_metrics()
        self.logger.info(
            'Received %s, pushed %s, dropped %s, sent %s, queue size %s',
            metrics['received_messages'],
            metrics['pushed_messages'],
            metrics['dropped_messages'],
            metrics['sent_messages'],
            metrics['queue_size'],
        )


class AdapterMetricsCollector(BaseMetricsCollector):
    def __init__(
        self,
        metrics_aggregator: MetricsAggregator,
        extra_labels: Dict[str, str],
    ):
        super().__init__(extra_labels)
        self._metrics_aggregator = metrics_aggregator
        self.register_metric(
            Counter('received_messages', 'Number of messages received by the adapter')
        )
        self.register_metric(
            Counter('pushed_messages', 'Number of messages pushed to the queue')
        )
        self.register_metric(
            Counter('dropped_messages', 'Number of messages dropped by the queue')
        )
        self.register_metric(
            Counter('sent_messages', 'Number of messages sent by the adapter')
        )
        self.register_metric(Gauge('queue_size', 'Number of messages in the queue'))

    def update_all_metrics(self):
        for k, v in self._metrics_aggregator.get_metrics().items():
            self._metrics[k].set(v)


def build_video_pipeline(config: Config):
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
        config.queue_path,
        config.queue_capacity * QUEUE_ITEM_SIZE,
    )
    # VideoPipeline is used to count passed frames
    pipeline = build_video_pipeline(config)
    ingress = Ingress(queue, config)
    egress = Egress(queue, pipeline, config)
    metrics_aggregator = MetricsAggregator(queue, ingress, egress)
    stats_logger = StatsLogger(metrics_aggregator, config)
    if config.metrics.provider is None:
        metrics_exporter = None
    elif config.metrics.provider == 'prometheus':
        metrics_exporter = PrometheusMetricsExporter(
            config.metrics.provider_params,
            AdapterMetricsCollector(
                metrics_aggregator,
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
