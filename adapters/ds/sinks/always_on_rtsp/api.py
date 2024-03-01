from enum import Enum
from fractions import Fraction
from http import HTTPStatus
from pathlib import Path
from threading import Thread
from typing import Dict, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from adapters.ds.sinks.always_on_rtsp.app_config import AppConfig
from adapters.ds.sinks.always_on_rtsp.config import (
    ENCODER_PROFILES,
    SUPPORTED_CODECS,
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
from adapters.ds.sinks.always_on_rtsp.utils import check_codec_is_available
from savant.gstreamer.codecs import CODEC_BY_NAME
from savant.utils.logging import get_logger

logger = get_logger('adapters.ao_sink.api')

SupportedCodecs = Enum('SupportedCodecs', {x.upper(): x for x in SUPPORTED_CODECS})


class OutputFormat(str, Enum):
    JSON = 'json'
    YAML = 'yaml'


class StreamModel(BaseModel):
    stub_file: Optional[Path] = None
    framerate: Optional[str] = Field(None, pattern=r'^\d+/\d+$', examples=['30/1'])
    bitrate: Optional[int] = Field(None, gt=0, examples=[4000000])
    profile: Optional[str] = None
    codec: Optional[SupportedCodecs] = None
    max_delay_ms: Optional[int] = Field(None, gt=0, examples=[1000])
    transfer_mode: Optional[TransferMode] = None
    rtsp_keep_alive: Optional[bool] = None
    metadata_output: Optional[MetadataOutput] = None
    sync_output: Optional[bool] = None

    def to_stream(self):
        codec = None
        if self.codec is not None:
            codec = CODEC_BY_NAME[self.codec.value]

        return Stream(
            stub_file=self.stub_file,
            framerate=self.framerate,
            codec=codec,
            bitrate=self.bitrate,
            profile=self.profile,
            max_delay_ms=self.max_delay_ms,
            transfer_mode=self.transfer_mode,
            rtsp_keep_alive=self.rtsp_keep_alive,
            metadata_output=self.metadata_output,
            sync_output=self.sync_output,
        )

    @staticmethod
    def from_stream(stream: Stream):
        return StreamModel(
            stub_file=stream.stub_file,
            framerate=stream.framerate,
            codec=stream.codec.value.name if stream.codec is not None else None,
            bitrate=stream.bitrate,
            profile=stream.profile,
            max_delay_ms=stream.max_delay_ms,
            transfer_mode=stream.transfer_mode,
            rtsp_keep_alive=stream.rtsp_keep_alive,
            metadata_output=stream.metadata_output,
            sync_output=stream.sync_output,
        )

    def to_dict(self):
        return {
            'stub_file': str(self.stub_file) if self.stub_file else None,
            'framerate': self.framerate,
            'codec': self.codec.value if self.codec else None,
            'bitrate': self.bitrate,
            'profile': self.profile,
            'max_delay_ms': self.max_delay_ms,
            'transfer_mode': self.transfer_mode.value if self.transfer_mode else None,
            'rtsp_keep_alive': self.rtsp_keep_alive,
            'metadata_output': (
                self.metadata_output.value if self.metadata_output else None
            ),
            'sync_output': self.sync_output,
        }


class StreamStatusModel(BaseModel):
    is_alive: bool
    exit_code: Optional[int]

    @staticmethod
    def from_stream(stream: Stream):
        return StreamStatusModel(
            is_alive=stream.exit_code is None,
            exit_code=stream.exit_code,
        )


class Api:
    def __init__(self, config: AppConfig, stream_manager: StreamManager):
        self._config = config
        self._stream_manager = stream_manager
        self._thread: Optional[Thread] = None
        self._app = FastAPI()
        self._app.get(
            '/streams',
            responses={200: {'content': {'application/x-yaml': {}}}},
        )(self.get_all_streams)
        self._app.get(
            '/streams/{source_id}',
            responses={200: {'content': {'application/x-yaml': {}}}},
        )(self.get_stream)
        self._app.put('/streams/{source_id}')(self.enable_stream)
        self._app.delete('/streams/{source_id}')(self.delete_stream)
        self._app.get('/status')(self.get_all_stream_statuses)
        self._app.get('/status/{source_id}')(self.get_stream_status)

    def get_all_streams(
        self,
        format: OutputFormat = OutputFormat.JSON,
    ) -> Dict[str, StreamModel]:
        response = {
            source_id: StreamModel.from_stream(stream)
            for source_id, stream in self._stream_manager.get_all_streams().items()
        }
        if format == OutputFormat.YAML:
            response_content = yaml.dump({k: v.to_dict() for k, v in response.items()})
            response = Response(
                content=response_content,
                media_type='application/x-yaml',
            )

        return response

    def get_stream(
        self,
        source_id: str,
        format: OutputFormat = OutputFormat.JSON,
    ) -> StreamModel:
        stream = self._stream_manager.get_stream(source_id)
        if stream is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f'Stream {source_id} not found.',
            )

        response = StreamModel.from_stream(stream)
        if format == OutputFormat.YAML:
            response_content = yaml.dump(response.to_dict())
            response = Response(
                content=response_content,
                media_type='application/x-yaml',
            )

        return response

    def enable_stream(self, source_id: str, stream: StreamModel) -> StreamModel:
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

        return StreamModel.from_stream(created_stream)

    def get_all_stream_statuses(self) -> Dict[str, StreamStatusModel]:
        return {
            source_id: StreamStatusModel.from_stream(stream)
            for source_id, stream in self._stream_manager.get_all_streams().items()
        }

    def get_stream_status(self, source_id: str) -> StreamStatusModel:
        stream = self._stream_manager.get_stream(source_id)
        if stream is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f'Stream {source_id} not found.',
            )

        return StreamStatusModel.from_stream(stream)

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
        logger.info('Starting API server on port %d', self._config.api_port)
        uvicorn.run(self._app, host='0.0.0.0', port=self._config.api_port)

    def start(self):
        self._thread = Thread(target=self.run_api, daemon=True)
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def validate_stream(self, stream: StreamModel):
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

        if stream.codec is not None:
            codec = CODEC_BY_NAME[stream.codec.value]
            if not check_codec_is_available(codec):
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f'Codec {stream.codec.value} is not available.',
                )
        else:
            codec = self._config.codec

        if stream.profile is not None:
            if stream.profile not in ENCODER_PROFILES[codec]:
                raise HTTPException(
                    status_code=HTTPStatus.BAD_REQUEST,
                    detail=f'Invalid profile {stream.profile} for codec {codec.value.name}.',
                )
