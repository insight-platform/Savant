from http import HTTPStatus
from threading import Thread
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from savant.utils.logging import get_logger

from . import LOGGER_PREFIX
from .config import Config
from .stream_manager import KvsStreamNotFoundError, StreamManager
from .stream_model import StreamModel

logger = get_logger(f'{LOGGER_PREFIX}.api')


class Api:
    """API server for the stream control API."""

    def __init__(self, config: Config, stream_manager: StreamManager):
        self.config = config
        self.stream_manager = stream_manager
        self.thread: Optional[Thread] = None
        self.app = FastAPI()
        self.app.get('/stream')(self.get_stream)
        self.app.put('/stream')(self.update_stream)
        self.app.put('/stream/play')(self.play_stream)
        self.app.put('/stream/stop')(self.stop_stream)

    def get_stream(self) -> StreamModel:
        """Get the current stream configuration."""

        stream = self.stream_manager.get_stream()
        if stream is None:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail='Stream not configured.',
            )

        return stream.without_credentials()

    def update_stream(self, stream: StreamModel) -> StreamModel:
        """Update stream configuration."""

        logger.info(
            'Updating stream configuration to: %r',
            stream.without_credentials(),
        )

        try:
            self.stream_manager.update_stream(stream)
        except KvsStreamNotFoundError as e:
            logger.warning('Stream not found: %s', e)
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f'Stream not found: {e}',
            )
        except Exception as e:
            logger.error('Error updating stream configuration: %s', e, exc_info=True)
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f'Failed to update stream configuration: {e}',
            )

        return self.get_stream()

    def play_stream(self) -> StreamModel:
        """Start the stream."""

        try:
            self.stream_manager.play_stream()
        except Exception as e:
            logger.error('Error starting the stream: %s', e, exc_info=True)
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f'Failed to start the stream: {e}',
            )

        return self.get_stream()

    def stop_stream(self) -> StreamModel:
        """Stop the stream."""

        try:
            self.stream_manager.stop_stream(stop_poller=False)
        except Exception as e:
            logger.error('Error stopping the stream: %s', e, exc_info=True)
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f'Failed to stop the stream: {e}',
            )

        return self.get_stream()

    def run_api(self):
        """Run the API server."""

        logger.info('Starting API server on port %d', self.config.api_port)
        uvicorn.run(self.app, host='0.0.0.0', port=self.config.api_port)

    def start(self):
        """Start the thread with the API server."""

        self.thread = Thread(target=self.run_api, daemon=True)
        self.thread.start()

    def is_running(self):
        """Check if the API server is running."""

        return self.thread.is_alive()
