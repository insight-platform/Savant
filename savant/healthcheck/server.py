from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

from savant.healthcheck.status import PipelineStatus
from savant.utils.logging import get_logger

logger = get_logger(__name__)

PIPELINE_RUNNING_STATUS_CONTENT = PipelineStatus.RUNNING.value.encode()


class HealthCheckHttpServer(HTTPServer):
    def __init__(self, host: str, port: int, http_path: str, status_filepath: Path):
        logger.info(
            'Starting healthcheck server at %s:%s. HTTP path: %s, status filepath: %s.',
            host,
            port,
            http_path,
            status_filepath,
        )
        self.http_path = http_path
        self.status_filepath = status_filepath
        super().__init__((host, port), HealthCheckRequestHandler)


class HealthCheckRequestHandler(BaseHTTPRequestHandler):
    server: HealthCheckHttpServer

    def do_GET(self) -> None:
        logger.debug('GET request received at path %s', self.path)
        if self.path != self.server.http_path:
            self._response(HTTPStatus.NOT_FOUND)
            return

        try:
            with self.server.status_filepath.open('rb') as f:
                content = f.read().strip()
        except Exception as e:
            logger.error(
                'Failed to read status file %s: %s',
                self.server.status_filepath,
                e,
            )
            self._response(HTTPStatus.SERVICE_UNAVAILABLE)
            return

        self._response(
            (
                HTTPStatus.OK
                if content == PIPELINE_RUNNING_STATUS_CONTENT
                else HTTPStatus.SERVICE_UNAVAILABLE
            ),
            content,
        )

    def _response(self, status: int, content: Optional[bytes] = None):
        self.send_response(status)
        self.send_header('Content-type', 'plain/text')
        self.end_headers()
        if content is not None:
            self.wfile.write(content)
