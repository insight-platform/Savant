import os
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen
from threading import Lock, Thread
from typing import Dict, Optional

from adapters.ds.sinks.always_on_rtsp.app_config import Config
from adapters.ds.sinks.always_on_rtsp.config import TransferMode
from adapters.ds.sinks.always_on_rtsp.utils import process_is_alive
from savant.utils.logging import get_logger

logger = get_logger('adapters.ao_sink.stream_manager')


@dataclass
class Stream:
    stub_file: Optional[Path] = None
    framerate: Optional[str] = None
    bitrate: Optional[int] = None
    profile: Optional[str] = None
    max_delay_ms: Optional[int] = None
    transfer_mode: Optional[TransferMode] = None
    rtsp_keep_alive: Optional[bool] = None
    metadata_output: Optional[bool] = None
    sync_output: Optional[bool] = None
    exit_code: Optional[int] = None


class StreamManager:
    def __init__(self, config: Config, stream_in_endpoint: str):
        self._config = config
        self._stream_in_endpoint = stream_in_endpoint
        self._streams: Dict[str, Stream] = {}
        self._processes: Dict[str, Popen] = {}
        self._is_running = False
        self._watcher_thread: Optional[Thread] = None
        self._lock = Lock()

    def start(self):
        logger.info('Starting stream manager')
        self._is_running = True
        self._watcher_thread = Thread(target=self.process_watcher, daemon=True)
        logger.info('Stream manager started')

    def stop(self):
        self._is_running = False

    def add_stream(self, source_id: str, stream: Stream):
        logger.info('Adding stream %r', source_id)
        with self._lock:
            if source_id in self._streams:
                raise ValueError(f'Stream with source_id {source_id!r} already exists.')
            # TODO: validate stream
            self._processes[source_id] = self.start_stream_process(source_id, stream)
            self._streams[source_id] = stream
        logger.info('Stream %r added', source_id)

    def start_stream_process(self, source_id: str, stream: Stream):
        process = Popen(
            ['python', '-m', 'adapters.ds.sinks.always_on_rtsp.ao_sink'],
            env={**os.environ, **self.build_envs(stream, source_id)},
        )
        logger.info(
            'Started Always-On-RTSP for source %r, PID: %s',
            source_id,
            process.pid,
        )
        if process.returncode is not None:
            raise RuntimeError(
                f'Failed to start Always-On-RTSP for source {source_id!r}. '
                f'Exit code: {process.returncode}.'
            )

        return process

    def build_envs(self, stream: Stream, source_id: str):
        rtsp_uri = f'{self._config.rtsp_uri.rstrip("/")}/{source_id}'
        envs = {
            'STUB_FILE_LOCATION': stream.stub_file,
            'FRAMERATE': stream.framerate,
            'ENCODER_BITRATE': stream.bitrate,
            'ENCODER_PROFILE': stream.profile,
            'MAX_DELAY_MS': stream.max_delay_ms,
            'TRANSFER_MODE': (
                stream.transfer_mode.value if stream.transfer_mode else None
            ),
            'RTSP_KEEP_ALIVE': stream.rtsp_keep_alive,
            'METADATA_OUTPUT': stream.metadata_output,
            'SYNC_OUTPUT': stream.sync_output,
            'SOURCE_ID': source_id,
            'RTSP_URI': rtsp_uri,
            'ZMQ_ENDPOINT': self._stream_in_endpoint,
        }
        envs = {k: str(v) for k, v in envs.items() if v is not None}

        return envs

    def process_watcher(self):
        while self._is_running:
            try:
                self.check_processes()
            except Exception as e:
                logger.error('Failed to check processes: %s', e, exc_info=True)
            time.sleep(1)  # TODO: configure

    def check_processes(self):
        with self._lock:
            if not self._is_running:
                return
            for source_id, process in list(self._processes.items()):
                exit_code = process_is_alive(process)
                if exit_code is not None:
                    logger.error(
                        'Always-On-RTSP for source %r exited. Exit code: %s.',
                        source_id,
                        exit_code,
                    )
                    self._streams[source_id].exit_code = exit_code

    def get_stream(self, source_id: str) -> Optional[Stream]:
        with self._lock:
            return self._streams.get(source_id)

    def get_all_streams(self) -> Dict[str, Stream]:
        with self._lock:
            return dict(self._streams)

    def delete_stream(self, source_id) -> int:
        logger.info('Deleting stream %r', source_id)
        with self._lock:
            if source_id not in self._streams:
                raise ValueError(f'Stream with source_id {source_id!r} does not exist.')
            process = self._processes[source_id]
            if process.returncode is None:
                logger.info('Terminating Always-On-RTSP for source %s', source_id)
                process.terminate()
                process.wait()
            exit_code = process.returncode
            del self._processes[source_id]
            del self._streams[source_id]
        logger.info('Stream %r deleted', source_id)

        return exit_code
