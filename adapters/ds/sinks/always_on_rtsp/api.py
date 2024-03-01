from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from adapters.ds.sinks.always_on_rtsp.app_config import Config
from adapters.ds.sinks.always_on_rtsp.config import TransferMode
from adapters.ds.sinks.always_on_rtsp.stream_manager import Stream, StreamManager
from savant.utils.logging import get_logger

logger = get_logger('adapters.ao_sink.api')


class OutputFormat(str, Enum):
    JSON = 'json'
    YAML = 'yaml'


class CreateStream(BaseModel):
    stub_file: Optional[Path] = None
    framerate: Optional[str] = None
    bitrate: Optional[int] = None
    profile: Optional[str] = None
    max_delay_ms: Optional[int] = None
    transfer_mode: Optional[TransferMode] = None
    rtsp_keep_alive: Optional[bool] = None
    metadata_output: Optional[bool] = None
    sync_output: Optional[bool] = None

    def to_stream(self):
        return Stream(
            stub_file=self.stub_file,
            framerate=self.framerate,
            bitrate=self.bitrate,
            profile=self.profile,
            max_delay_ms=self.max_delay_ms,
            transfer_mode=self.transfer_mode,
            rtsp_keep_alive=self.rtsp_keep_alive,
            metadata_output=self.metadata_output,
            sync_output=self.sync_output,
        )


class ResponseStream(CreateStream):
    source_id: str
    exit_code: Optional[int] = None

    @staticmethod
    def from_stream(source_id: str, stream: Stream):
        return ResponseStream(
            source_id=source_id,
            stub_file=stream.stub_file,
            framerate=stream.framerate,
            bitrate=stream.bitrate,
            profile=stream.profile,
            max_delay_ms=stream.max_delay_ms,
            transfer_mode=stream.transfer_mode,
            rtsp_keep_alive=stream.rtsp_keep_alive,
            metadata_output=stream.metadata_output,
            sync_output=stream.sync_output,
            exit_code=stream.exit_code,
        )


class Api:
    def __init__(self, config: Config, stream_manager: StreamManager):
        self._config = config
        self._stream_manager = stream_manager
        self._thread: Optional[Thread] = None
        self._app = FastAPI()
        self._app.get('/streams/{output_format}')(self.get_streams)
        self._app.put('/streams/{source_id}')(self.enable_stream)
        self._app.delete('/streams/{source_id}')(self.delete_stream)

    def get_streams(self, output_format: OutputFormat):
        # TODO: support YAML output format
        return [
            ResponseStream.from_stream(source_id, stream)
            for source_id, stream in self._stream_manager.get_all_streams().items()
        ]

    def enable_stream(self, source_id: str, stream: CreateStream):
        try:
            self._stream_manager.add_stream(source_id, stream.to_stream())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        created_stream = self._stream_manager.get_stream(source_id)

        return ResponseStream.from_stream(source_id, created_stream)

    def delete_stream(self, source_id: str):
        try:
            self._stream_manager.delete_stream(source_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return 'ok'

    def run_api(self):
        uvicorn.run(self._app, host='0.0.0.0', port=5000)

    def start(self):
        self._thread = Thread(target=self.run_api, daemon=True)
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()
