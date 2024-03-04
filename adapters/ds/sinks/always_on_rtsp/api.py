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


class StreamStatusModel(BaseModel):
    """Status of a stream."""

    is_alive: bool = Field(
        description='Whether the stream is alive.',
    )
    exit_code: Optional[int] = Field(
        None,
        description='Exit code of the stream process.',
    )

    @staticmethod
    def from_stream(stream: Stream):
        """Build a StreamStatusModel from a Stream object."""

        return StreamStatusModel(
            is_alive=stream.exit_code is None,
            exit_code=stream.exit_code,
        )

    def to_dict(self):
        """Convert the model to a dictionary."""

        return {
            'is_alive': self.is_alive,
            'exit_code': self.exit_code,
        }


class StreamModel(BaseModel):
    """Stream configuration."""

    stub_file: Optional[Path] = Field(
        None,
        description='Location of the stub image file. Image file must be in JPEG format.',
        examples=['/stub_imgs/smpte100_1280x720.jpeg'],
    )
    framerate: Optional[str] = Field(
        None,
        description='Frame rate of the output stream.',
        pattern=r'^\d+/\d+$',
        examples=['30/1'],
    )
    bitrate: Optional[int] = Field(
        None,
        description='Encoding bitrate in bit/s.',
        gt=0,
        examples=[4000000],
    )
    profile: Optional[str] = Field(
        None,
        description=(
            'Encoding profile. '
            'For "h264" one of: "Baseline", "Main", "High". '
            'For "hevc" one of: "Main", "Main10", "FREXT".'
        ),
    )
    codec: Optional[SupportedCodecs] = Field(
        None,
        description='Encoding codec.',
    )
    max_delay_ms: Optional[int] = Field(
        None,
        description='Maximum delay for the last frame in milliseconds.',
        gt=0,
        examples=[1000],
    )
    latency_ms: Optional[int] = Field(
        None,
        description='Amount of ms to buffer RTSP stream.',
        gt=0,
        examples=[100],
    )
    transfer_mode: Optional[TransferMode] = Field(
        None,
        description='Transfer mode.',
    )
    rtsp_keep_alive: Optional[bool] = Field(
        None,
        description='Send RTSP keep alive packets, disable for old incompatible server.',
    )
    metadata_output: Optional[MetadataOutput] = Field(
        None,
        description='Where to dump metadata.',
    )
    sync_output: Optional[bool] = Field(
        None,
        description=(
            'Show frames on sink synchronously (i.e. at the source file rate). '
            'Note: inbound stream is not stable with this flag, try to avoid it.'
        ),
    )
    status: Optional[StreamStatusModel] = Field(
        None,
        description='Status of the stream.',
    )

    def to_stream(self):
        """Convert the model to a Stream object."""

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
            latency_ms=self.latency_ms,
            transfer_mode=self.transfer_mode,
            rtsp_keep_alive=self.rtsp_keep_alive,
            metadata_output=self.metadata_output,
            sync_output=self.sync_output,
        )

    @staticmethod
    def from_stream(stream: Stream):
        """Build a StreamModel from a Stream object."""

        return StreamModel(
            stub_file=stream.stub_file,
            framerate=stream.framerate,
            codec=stream.codec.value.name if stream.codec is not None else None,
            bitrate=stream.bitrate,
            profile=stream.profile,
            max_delay_ms=stream.max_delay_ms,
            latency_ms=stream.latency_ms,
            transfer_mode=stream.transfer_mode,
            rtsp_keep_alive=stream.rtsp_keep_alive,
            metadata_output=stream.metadata_output,
            sync_output=stream.sync_output,
            status=StreamStatusModel.from_stream(stream),
        )

    def to_dict(self):
        """Convert the model to a dictionary."""

        return {
            'stub_file': str(self.stub_file) if self.stub_file else None,
            'framerate': self.framerate,
            'codec': self.codec.value if self.codec else None,
            'bitrate': self.bitrate,
            'profile': self.profile,
            'max_delay_ms': self.max_delay_ms,
            'latency_ms': self.latency_ms,
            'transfer_mode': self.transfer_mode.value if self.transfer_mode else None,
            'rtsp_keep_alive': self.rtsp_keep_alive,
            'metadata_output': (
                self.metadata_output.value if self.metadata_output else None
            ),
            'sync_output': self.sync_output,
            'status': self.status.to_dict() if self.status else None,
        }


class Api:
    """API server for the stream control API."""

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
        self._app.put('/streams/{source_id}')(self.create_stream)
        self._app.delete('/streams/{source_id}')(self.delete_stream)

    def get_all_streams(
        self,
        format: OutputFormat = OutputFormat.JSON,
    ) -> Dict[str, StreamModel]:
        """List all configured streams."""

        response = {
            source_id: StreamModel.from_stream(stream)
            for source_id, stream in self._stream_manager.get_all_streams().items()
        }
        if format == OutputFormat.YAML:
            response_content = yaml.dump(
                {k: v.to_dict() for k, v in response.items()},
                sort_keys=False,
            )
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
        """Get a stream by source ID."""

        stream = self._stream_manager.get_stream(source_id)
        if stream is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f'Stream {source_id} not found.',
            )

        response = StreamModel.from_stream(stream)
        if format == OutputFormat.YAML:
            response_content = yaml.dump(response.to_dict(), sort_keys=False)
            response = Response(
                content=response_content,
                media_type='application/x-yaml',
            )

        return response

    def create_stream(self, source_id: str, stream: StreamModel) -> StreamModel:
        """Create a new stream and start it."""

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

    def delete_stream(self, source_id: str):
        """Stop and delete a stream."""

        try:
            self._stream_manager.delete_stream(source_id)
        except StreamNotFoundError as e:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f'Stream {e.source_id} not found.',
            )

        return 'ok'

    def run_api(self):
        """Run the API server."""

        logger.info('Starting API server on port %d', self._config.api_port)
        uvicorn.run(self._app, host='0.0.0.0', port=self._config.api_port)

    def start(self):
        """Start the thread with the API server."""

        self._thread = Thread(target=self.run_api, daemon=True)
        self._thread.start()

    def is_alive(self):
        """Check if the API server is running."""

        return self._thread.is_alive()

    def validate_stream(self, stream: StreamModel):
        """Validate a stream configuration."""

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
