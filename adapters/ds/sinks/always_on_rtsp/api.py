from fractions import Fraction
from http import HTTPStatus
from pathlib import Path
from threading import Thread
from typing import List, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from adapters.ds.sinks.always_on_rtsp.app_config import Config
from adapters.ds.sinks.always_on_rtsp.config import (
    ENCODER_PROFILES,
    MetadataOutput,
    TransferMode,
)
from adapters.ds.sinks.always_on_rtsp.stream_manager import (
    FailedToStartStreamError,
    Stream,
    StreamAlreadyExistsError,
    StreamManager,
    StreamNotFoundError,
)
from savant.utils.logging import get_logger

logger = get_logger('adapters.ao_sink.api')


class CreateStream(BaseModel):
    stub_file: Optional[Path] = None
    framerate: Optional[str] = Field(None, pattern=r'^\d+/\d+$', examples=['30/1'])
    bitrate: Optional[int] = Field(None, gt=0, examples=[4000000])
    profile: Optional[str] = None
    max_delay_ms: Optional[int] = Field(None, gt=0, examples=[1000])
    transfer_mode: Optional[TransferMode] = None
    rtsp_keep_alive: Optional[bool] = None
    metadata_output: Optional[MetadataOutput] = None
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
        self._app.get('/streams/json')(self.get_streams_json)
        self._app.get(
            '/streams/yaml',
            responses={200: {'content': {'application/x-yaml': {}}}},
        )(self.get_streams_yaml)
        self._app.put('/streams/{source_id}')(self.enable_stream)
        self._app.delete('/streams/{source_id}')(self.delete_stream)

    def get_streams_json(self) -> List[ResponseStream]:
        return [
            ResponseStream.from_stream(source_id, stream)
            for source_id, stream in self._stream_manager.get_all_streams().items()
        ]

    def get_streams_yaml(self):
        response = yaml.dump([x.dict() for x in self.get_streams_json()])

        return Response(content=response, media_type='application/x-yaml')

    def enable_stream(self, source_id: str, stream: CreateStream) -> ResponseStream:
        self.validate_stream(stream)
        try:
            self._stream_manager.add_stream(source_id, stream.to_stream())
        except StreamAlreadyExistsError:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f'Stream {source_id} already exists.',
            )
        except FailedToStartStreamError as e:
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f'Failed to start stream {source_id}. Exit code: {e.exit_code}.',
            )

        created_stream = self._stream_manager.get_stream(source_id)

        return ResponseStream.from_stream(source_id, created_stream)

    def delete_stream(self, source_id: str):
        try:
            self._stream_manager.delete_stream(source_id)
        except StreamNotFoundError as e:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f'Stream {e.source_id} not found.',
            )

        return 'ok'

    def run_api(self):
        uvicorn.run(self._app, host='0.0.0.0', port=self._config.api_port)

    def start(self):
        self._thread = Thread(target=self.run_api, daemon=True)
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def validate_stream(self, stream: CreateStream):
        if stream.stub_file:
            if not stream.stub_file.exists():
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f'Stub file {stream.stub_file} does not exist.',
                )
            if not stream.stub_file.is_file():
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f'Stub file {stream.stub_file} is not a file.',
                )
        if stream.framerate is not None:
            try:
                Fraction(stream.framerate)
            except Exception:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f'Invalid framerate {stream.framerate}.',
                )

        if stream.profile is not None:
            if stream.profile not in ENCODER_PROFILES[self._config.codec]:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f'Invalid profile {stream.profile} for codec {self._config.codec.value.name}.',
                )
