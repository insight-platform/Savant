from datetime import datetime
from threading import Lock
from typing import Optional

from botocore.exceptions import ClientError

from savant.utils.logging import get_logger

from . import LOGGER_PREFIX
from .config import Config
from .pipeline import Pipeline
from .poller import FragmentsPoller
from .state import State
from .stream_model import AwsCredentials, StreamModel

logger = get_logger(f'{LOGGER_PREFIX}.stream_manager')

THREAD_JOIN_TIMEOUT = 10


class StreamManagerError(Exception):
    """Stream manager error."""


class KvsStreamError(StreamManagerError):
    """KVS stream error."""


class KvsStreamNotFoundError(KvsStreamError):
    """KVS stream not found error."""


class StreamManager:
    """Stream manager."""

    def __init__(
        self,
        config: Config,
        thread_join_timeout: float = THREAD_JOIN_TIMEOUT,
    ):
        self.config = config
        self.thread_join_timeout = thread_join_timeout
        self.poller: Optional[FragmentsPoller] = None
        self.pipeline: Optional[Pipeline] = None
        self.stream: Optional[StreamModel] = None
        self.state: Optional[State] = None
        self.lock = Lock()

    def update_stream(self, stream: StreamModel):
        """Update the stream configuration."""

        logger.info('Updating stream %r/%r', stream.name, stream.source_id)
        if not self.need_update(stream):
            logger.info(
                'Stream %r/%r configuration not changed',
                self.stream.name,
                self.stream.source_id,
            )
            return

        need_poller_update = self.need_poller_update(stream)
        self.fill(stream)

        with self.lock:
            if need_poller_update:
                if stream.is_playing:
                    next_poller = self.start_poller(stream)
                else:
                    next_poller = None
            else:
                next_poller = self.poller
            self.stop_stream(stop_poller=need_poller_update)
            self.stream = stream
            if self.state is not None:
                self.state.update(state=stream)
            self.poller = next_poller
            if stream.is_playing:
                self.pipeline = self.start_pipeline(stream, next_poller)
            logger.info(
                'Stream updated to %s/%s', self.stream.name, self.stream.source_id
            )

    def start_poller(self, stream: StreamModel):
        """Start the fragments poller."""

        try:
            poller = FragmentsPoller(stream)
            poller.start()
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning(
                    'Failed to start fragments poller for stream %s: %s',
                    stream.name,
                    e,
                )
                raise KvsStreamNotFoundError(stream.name) from e

            raise KvsStreamError(f'Failed to start fragments poller: {e}') from e

        return poller

    def start_pipeline(self, stream: StreamModel, poller: FragmentsPoller):
        """Start the pipeline."""

        pipeline = Pipeline(
            config=self.config,
            stream=stream,
            queue=poller.queue,
            state=self.state,
        )
        pipeline.start()

        return pipeline

    def play_stream(self):
        """Start playing the stream."""

        if self.stream is None:
            return

        with self.lock:
            if self.stream.is_playing:
                return

            if self.poller is None:
                self.poller = self.start_poller(self.stream)
            if self.pipeline is None:
                self.pipeline = self.start_pipeline(self.stream, self.poller)
            self.stream.is_playing = True

    def stop_stream(self, stop_poller: bool):
        """Stop the fragments poller and the pipeline."""

        self.stream.is_playing = False

        if stop_poller and self.poller is not None:
            logger.info('Stopping fragments poller')
            self.poller.stop()

        if self.pipeline is not None:
            logger.info('Stopping pipeline')
            self.pipeline.stop()

        if stop_poller and self.poller is not None:
            self.poller.join(self.thread_join_timeout)
            self.poller = None
            logger.info('Fragments poller stopped')

        if self.pipeline is not None:
            self.pipeline.join(self.thread_join_timeout)
            self.update_stream_ts()
            self.pipeline = None
            logger.info('Pipeline stopped')

    def is_running(self):
        """Check if the stream manager is running."""

        with self.lock:
            if self.stream is None or not self.stream.is_playing:
                return True

            if self.poller is None or not self.poller.is_running:
                return False

            if self.pipeline is None or not self.pipeline.is_running:
                return False

        return True

    def start(self):
        """Start the stream manager and the initial stream."""

        with self.lock:
            if self.config.save_state:
                self.state = State(self.config.state_path)
                self.stream = self.state.get_state()
            if self.stream is None:
                self.stream: Optional[StreamModel] = StreamModel(
                    name=self.config.stream_name,
                    source_id=self.config.source_id,
                    timestamp=self.config.timestamp,
                    credentials=AwsCredentials(
                        region=self.config.aws.region,
                        access_key=self.config.aws.access_key,
                        secret_key=self.config.aws.secret_key,
                    ),
                    is_playing=self.config.playing,
                )
                if self.state is not None:
                    self.state.update(state=self.stream)

            if self.state is not None:
                self.state.start()
            if self.stream.is_playing is None or self.stream.is_playing:
                self.poller = self.start_poller(self.stream)
                self.pipeline = self.start_pipeline(
                    self.stream,
                    self.poller,
                )
                self.stream.is_playing = True
        logger.info('Stream manager started')

    def stop(self):
        """Stop the stream manager and the current stream."""

        with self.lock:
            self.stop_stream(stop_poller=True)
            self.stream = None
            if self.state is not None:
                self.state.stop()
        logger.info('Stream manager stopped')

    def fill(self, stream: StreamModel):
        """Fill the missing stream configurations with the current values."""

        if stream.name is None:
            stream.name = self.stream.name
        if stream.source_id is None:
            stream.source_id = self.stream.source_id
        if stream.credentials is None:
            stream.credentials = self.stream.credentials
        if stream.timestamp is None:
            last_ts = self.pipeline.last_ts if self.pipeline is not None else None
            if last_ts is not None:
                stream.timestamp = datetime.utcfromtimestamp(last_ts)
            else:
                stream.timestamp = self.stream.timestamp
        if stream.is_playing is None:
            stream.is_playing = self.stream.is_playing

    def get_stream(self):
        """Get the current stream configuration."""

        with self.lock:
            if self.stream is None:
                return None
            self.update_stream_ts()

            return self.stream

    def need_update(self, stream: StreamModel) -> bool:
        """Check if the stream configuration needs to be updated."""

        if self.stream is None:
            return True

        if stream.name is not None and stream.name != self.stream.name:
            return True

        if stream.source_id is not None and stream.source_id != self.stream.source_id:
            return True

        if (
            stream.credentials is not None
            and stream.credentials != self.stream.credentials
        ):
            return True

        if (
            stream.is_playing is not None
            and stream.is_playing != self.stream.is_playing
        ):
            return True

        return stream.timestamp is not None

    def need_poller_update(self, stream: StreamModel) -> bool:
        """Check if the fragments poller needs to be updated."""

        if stream.name is not None and stream.name != self.stream.name:
            return True

        if (
            stream.credentials is not None
            and stream.credentials != self.stream.credentials
        ):
            return True

        return stream.timestamp is not None

    def update_stream_ts(self):
        if self.pipeline is None:
            return
        last_ts = self.pipeline.last_ts
        if last_ts is None:
            return
        self.stream.timestamp = datetime.utcfromtimestamp(last_ts)
