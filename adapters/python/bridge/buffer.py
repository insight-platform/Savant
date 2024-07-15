import json
import os
import signal
import time
from typing import Dict, Optional, Tuple

import msgpack
from rocksq.blocking import PersistentQueueWithCapacity
from savant_rs.pipeline2 import (
    StageFunction,
    VideoPipeline,
    VideoPipelineConfiguration,
    VideoPipelineStagePayloadType,
)
from savant_rs.primitives import VideoFrame, VideoFrameContent
from savant_rs.utils.serialization import (
    Message,
    load_message_from_bytes,
    save_message_to_bytes,
)
from savant_rs.zmq import BlockingWriter, WriterConfigBuilder, WriterSocketType, WriterResultSuccess, WriterResultAck

from adapters.shared.thread import BaseThreadWorker
from savant.metrics import Counter, Gauge
from savant.metrics.prometheus import BaseMetricsCollector, PrometheusMetricsExporter
from savant.utils.config import opt_config, req_config, strtobool
from savant.utils.logging import get_logger, init_logging
from savant.utils.welcome import get_starting_message
from savant.utils.zeromq import ZeroMQMessage, ZeroMQSource

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


class MessageDumpConfig:
    """Message dump configuration for the adapter."""

    def __init__(self):
        self.enabled = opt_config('MESSAGE_DUMP_ENABLED', False, strtobool)
        self.path = opt_config('MESSAGE_DUMP_PATH', '/tmp/buffer-adapter-dump', str)
        self.segment_duration = opt_config('MESSAGE_DUMP_SEGMENT_DURATION', 60, int)
        self.segment_template = opt_config(
            'MESSAGE_DUMP_SEGMENT_TEMPLATE', 'dump-%Y-%m-%d-%H-%M-%S.msgpack', str
        )


class BufferConfig:
    """Buffer configuration for the adapter."""

    def __init__(self):
        self.path = req_config('BUFFER_PATH')
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
        self.zmq_src_endpoint = req_config('ZMQ_SRC_ENDPOINT')
        self.zmq_sink_endpoint = req_config('ZMQ_SINK_ENDPOINT')
        self.buffer = BufferConfig()
        self.message_dump = MessageDumpConfig()
        self.idle_polling_period = opt_config('IDLE_POLLING_PERIOD', 0.005, float)
        self.stats_log_interval = opt_config('STATS_LOG_INTERVAL', 60, int)
        self.metrics = MetricsConfig()


class MessageDumper:
    """Message dump for the adapter."""

    def __init__(self, config: MessageDumpConfig):
        self._config = config

        # create the dump directory if it does not exist and it is enabled
        if self._config.enabled and not os.path.exists(self._config.path):
            os.makedirs(self._config.path)

        self._segment_start = 0
        self._segment_path = self._get_segment_path()
        self._file = None

    def dump(self, message: ZeroMQMessage):
        """Dump the message to the message dump."""

        if not self._config.enabled:
            logger.trace('Message dump is not enabled. Skipping')
            return

        if time.time() - self._segment_start > self._config.segment_duration:
            self._segment_start = time.time()
            self._segment_path = self._get_segment_path()
            logger.info(
                'Rotating message dump segment. New segment: %s', self._segment_path
            )
            if self._file is not None:
                self._file.close()
            self._file = open(self._segment_path, 'wb')

        topic = message.topic
        meta = message.message
        content = message.content
        ts = time.time_ns()
        self._file.write(
            msgpack.packb(
                (ts, topic, save_message_to_bytes(meta), content), use_bin_type=True
            )
        )

    def _get_segment_path(self):
        return os.path.join(
            self._config.path,
            time.strftime(self._config.segment_template, time.gmtime()),
        )


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
        self._message_dumper = MessageDumper(config.message_dump)
        self._buffer_is_full = False
        self._received_messages = 0
        self._last_received_message = 0
        self._pushed_messages = 0
        self._last_pushed_message = 0
        self._dropped_messages = 0
        self._last_dropped_message = 0
        self._zmq_source = ZeroMQSource(config.zmq_src_endpoint)

    def workload(self):
        self.logger.info('Starting Ingress')
        self._zmq_source.start()
        while self.is_running:
            try:
                zmq_message = self._zmq_source.next_message()
                if zmq_message is not None:
                    self.logger.debug('Received message from the source ZeroMQ socket')
                    self.handle_next_message(zmq_message)
            except Exception as e:
                self.logger.error('Failed to poll message: %s', e)
                self.is_running = False
                break
        self._zmq_source.terminate()
        self.logger.info('Ingress was stopped')

    def handle_next_message(self, message: ZeroMQMessage):
        """Handle the next message from the source ZeroMQ socket."""
        self._message_dumper.dump(message)
        self._received_messages += 1
        self._last_received_message = time.time()
        if message.message.is_video_frame():
            pushed = self.push_frame(message)
        else:
            pushed = self.push_service_message(message)
        if pushed:
            self._pushed_messages += 1
            self._last_pushed_message = time.time()
        else:
            self._dropped_messages += 1
            self._last_dropped_message = time.time()

    def push_frame(self, message: ZeroMQMessage) -> bool:
        """Push frame to the buffer."""

        buffer_size = self._queue.len
        if self._buffer_is_full:
            if buffer_size >= self._config.buffer.threshold:
                self.logger.debug('Buffer is full, dropping the frame')
                return False

            self._buffer_is_full = False

        elif buffer_size + QUEUE_ITEM_SIZE >= self._config.buffer.len:
            self._buffer_is_full = True
            self.logger.debug('Buffer is full, dropping the frame')
            return False

        self._push_message(message)
        self.logger.debug('Pushed frame to the buffer')
        return True

    def push_service_message(self, message: ZeroMQMessage) -> bool:
        """Push service message to the buffer."""

        try:
            self._push_message(message)
        except Exception as e:
            if e.args[0] != 'Failed to push item: Queue is full':
                raise
            self.logger.debug('Buffer is full, dropping the message')
            return False

        self.logger.debug('Pushed message to the buffer')
        return True

    def _push_message(self, message: ZeroMQMessage):
        message_parts = [
            bytes(message.topic),
            save_message_to_bytes(message.message),
            message.content,
        ]
        self._queue.push(message_parts)

    @property
    def received_messages(self) -> int:
        """Number of messages received from the source ZeroMQ socket."""
        return self._received_messages

    @property
    def last_received_message(self) -> int:
        """Timestamp of the last received message."""
        return self._last_received_message

    @property
    def pushed_messages(self) -> int:
        """Number of messages pushed to the buffer."""
        return self._pushed_messages

    @property
    def last_pushed_message(self) -> int:
        """Timestamp of the last pushed message."""
        return self._last_pushed_message

    @property
    def dropped_messages(self) -> int:
        """Number of messages dropped by the adapter."""
        return self._dropped_messages

    @property
    def last_dropped_message(self) -> int:
        """Timestamp of the last dropped message."""
        return self._last_dropped_message


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
        self._last_sent_message = 0
        self._pipeline = pipeline
        self._video_frame = VideoFrame(
            source_id='test',
            framerate='30/1',
            width=1280,
            height=720,
            content=VideoFrameContent.none(),
        )

        config_builder = WriterConfigBuilder(config.zmq_sink_endpoint)
        config = config_builder.build()
        assert (
            config.socket_type == WriterSocketType.Dealer
        ), 'Only DEALER socket type is supported for Egress'
        self._write_timeout_send_retries = config.send_retries + 1

        self._writer = BlockingWriter(config)

    def workload(self):
        self.logger.info('Starting Egress')
        self._writer.start()
        while self.is_running:
            try:
                message = self.pop_next_message()
                if message is not None:
                    is_sent = False
                    while not is_sent:
                        self.logger.debug('Sending message to the sink ZeroMQ socket')
                        send_message_result = self._writer.send_message(*message)
                        if isinstance(send_message_result, (WriterResultSuccess, WriterResultAck)):
                            self._sent_messages += 1
                            self._last_sent_message = time.time()
                            is_sent = True
                        else:
                            self.logger.warning('Failed to send message to the sink ZeroMQ socket: %s. '
                                                'Retrying', send_message_result)
            except Exception as e:
                self.logger.error('Failed to send message: %s', e)
                self.is_running = False
                break
        self._writer.shutdown()
        self.logger.info('Egress was stopped')

    def pop_next_message(self) -> Optional[Tuple[str, Message, bytes]]:
        """Pop the next message from the buffer.

        When the buffer is empty, wait until it is not empty and then pop the message.
        """

        if self._queue.len < QUEUE_ITEM_SIZE:
            self.logger.trace(
                'Buffer is empty, waiting for %s seconds',
                self._idle_polling_period,
            )
            time.sleep(self._idle_polling_period)
            return None

        topic, message, data = self._queue.pop(QUEUE_ITEM_SIZE)
        topic = topic.decode()
        message = load_message_from_bytes(message)

        frame_id = self._pipeline.add_frame('fps-meter', self._video_frame)
        self._pipeline.delete(frame_id)

        return topic, message, data

    @property
    def last_sent_message(self) -> int:
        """Timestamp of the last sent message."""
        return self._last_sent_message

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
        last_received_message = self._ingress.last_received_message
        pushed_messages = self._ingress.pushed_messages
        last_pushed_message = self._ingress.last_pushed_message
        dropped_messages = self._ingress.dropped_messages
        last_dropped_message = self._ingress.last_dropped_message
        sent_messages = self._egress.sent_messages
        last_sent_message = self._egress.last_sent_message
        buffer_size = self._queue.len // QUEUE_ITEM_SIZE
        payload_size = self._queue.payload_size

        return {
            'received_messages': received_messages,
            'pushed_messages': pushed_messages,
            'dropped_messages': dropped_messages,
            'sent_messages': sent_messages,
            'buffer_size': buffer_size,
            'payload_size': payload_size,
            'last_received_message': last_received_message,
            'last_pushed_message': last_pushed_message,
            'last_dropped_message': last_dropped_message,
            'last_sent_message': last_sent_message,
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
            'Received %s, pushed %s, dropped %s, sent %s, buffer size %s, payload size %s, last message received %s, last message pushed %s, last message dropped %s, last message sent %s',
            stats['received_messages'],
            stats['pushed_messages'],
            stats['dropped_messages'],
            stats['sent_messages'],
            stats['buffer_size'],
            stats['payload_size'],
            stats['last_received_message'],
            stats['last_pushed_message'],
            stats['last_dropped_message'],
            stats['last_sent_message'],
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
        self.register_metric(
            Gauge('payload_size', 'Size of the queue in bytes (only payload)')
        )
        self.register_metric(
            Gauge('last_received_message', 'Timestamp of the last received message')
        )
        self.register_metric(
            Gauge('last_pushed_message', 'Timestamp of the last pushed message')
        )
        self.register_metric(
            Gauge('last_dropped_message', 'Timestamp of the last dropped message')
        )
        self.register_metric(
            Gauge('last_sent_message', 'Timestamp of the last sent message')
        )

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
        [
            (
                'fps-meter',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            )
        ],
        conf,
    )


def main():
    init_logging()
    logger.info(get_starting_message('buffer bridge adapter'))
    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))
    config = Config()
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
