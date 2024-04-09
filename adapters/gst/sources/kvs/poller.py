from dataclasses import dataclass
from queue import Full, Queue
from threading import Event
from typing import Optional, Union

import boto3
from amazon_kinesis_video_consumer_library.kinesis_video_fragment_processor import (
    KvsFragementProcessor,
)
from amazon_kinesis_video_consumer_library.kinesis_video_streams_parser import (
    KvsConsumerLibrary,
)

from adapters.shared.thread import BaseThreadWorker

from . import LOGGER_PREFIX
from .stream_model import StreamModel

MAX_QUEUE_SIZE = 10
QUEUE_PUT_TIMEOUT = 1


@dataclass
class Fragment:
    """Fragment from KVS stream."""

    fragment_number: str
    timestamp: float
    content: bytearray


class FragmentsPoller(BaseThreadWorker):
    """Polls fragments from KVS stream and puts them into a queue."""

    def __init__(
        self,
        stream: StreamModel,
        max_queue_size: int = MAX_QUEUE_SIZE,
        queue_put_timeout: float = QUEUE_PUT_TIMEOUT,
    ):
        super().__init__(
            f'FragmentsPoller-{stream.name}',
            logger_name=f'{LOGGER_PREFIX}.poller',
        )
        self.stream = stream
        self.queue: Queue[Fragment] = Queue(max_queue_size)
        self.queue_put_timeout = queue_put_timeout
        self.last_fragment_number: Optional[str] = None

        self.event = Event()
        self.fragment_processor = KvsFragementProcessor()
        self.session = boto3.Session(
            region_name=stream.credentials.region,
            aws_access_key_id=stream.credentials.access_key,
            aws_secret_access_key=stream.credentials.secret_key,
        )
        self.kvs_client = self.session.client('kinesisvideo')
        self.endpoint = self.kvs_client.get_data_endpoint(
            StreamName=self.stream.name,
            APIName='GET_MEDIA',
        ).get('DataEndpoint')
        if self.endpoint is None:
            self.set_error(f'Failed to get endpoint for stream {self.stream.name}')
            return
        self.logger.debug('Endpoint for stream %s: %s', self.stream.name, self.endpoint)
        self.media_client = self.session.client(
            'kinesis-video-media',
            endpoint_url=self.endpoint,
        )
        self.consumer = self.create_consumer()

    def workload(self):
        self.logger.info(
            'Starting polling for fragments from stream %s', self.stream.name
        )

        while self.is_running:
            try:
                self.workload_loop()
            except Exception as e:
                self.is_running = False
                self.set_error(
                    f'Error polling for fragments from stream {self.stream.name}: {e}'
                )
                return

        self.logger.info(
            'Stopped polling for fragments from stream %s', self.stream.name
        )

    def workload_loop(self):
        if self.consumer is None:
            self.consumer = self.create_consumer()
        self.consumer.start()
        self.event.wait()
        self.consumer.stop_thread()
        self.consumer = None

    def create_consumer(self):
        """Create KVS consumer for the stream."""

        response = self.media_client.get_media(
            StreamName=self.stream.name,
            StartSelector=self.next_stream_selector(),
        )
        self.event.clear()
        consumer = KvsConsumerLibrary(
            self.stream.name,
            response,
            self.on_fragment_arrived,
            self.on_stream_read_complete,
            self.on_stream_read_exception,
        )

        return consumer

    def on_fragment_arrived(
        self,
        stream_name: str,
        fragment_bytes: bytearray,
        fragment_dom,
        fragment_receive_duration: float,
    ):
        """Callback for fragment arrival. Puts fragment into the queue."""

        if not self.is_running:
            return

        self.logger.debug(
            'Received fragment from stream %r with length %s. Receive duration: %s.',
            stream_name,
            len(fragment_bytes),
            fragment_receive_duration,
        )
        fragment_tags = self.fragment_processor.get_fragment_tags(fragment_dom)
        fragment_number = fragment_tags.get('AWS_KINESISVIDEO_FRAGMENT_NUMBER')
        timestamp = float(fragment_tags.get('AWS_KINESISVIDEO_PRODUCER_TIMESTAMP'))
        self.logger.debug(
            'Processing fragment %r with timestamp %s from stream %r.',
            fragment_number,
            timestamp,
            stream_name,
        )
        while self.is_running:
            try:
                self.queue.put(
                    Fragment(fragment_number, timestamp, fragment_bytes),
                    timeout=self.queue_put_timeout,
                )
                self.last_fragment_number = fragment_number
                self.logger.debug(
                    'Processed fragment %r with timestamp %s from stream %r.',
                    fragment_number,
                    timestamp,
                    stream_name,
                )
                break
            except Full:
                self.logger.debug('Queue is full. Waiting for space.')

    def on_stream_read_complete(self, stream_name: str):
        """Callback for stream read completion. Sets event to stop the consumer."""

        self.logger.info('Stream %r read complete.', stream_name)
        self.event.set()

    def on_stream_read_exception(self, stream_name: str, error: Union[str, Exception]):
        """Callback for stream read exception. Sets error and event to stop the consumer."""

        self.set_error(f'Error reading stream {stream_name}: {error}')
        self.event.set()

    def next_stream_selector(self):
        """Get selector for the next fragment."""

        if self.last_fragment_number is None:
            return {
                'StartSelectorType': 'PRODUCER_TIMESTAMP',
                'StartTimestamp': self.stream.timestamp,
            }

        return {
            'StartSelectorType': 'FRAGMENT_NUMBER',
            'AfterFragmentNumber': self.last_fragment_number,
        }

    def stop(self):
        """Stop polling for fragments."""

        super().stop()
        self.event.set()
