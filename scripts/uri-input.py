#!/usr/bin/env python3
import mimetypes
import os
import string
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import click

from savant.utils.file_types import FileType, parse_mime_types
from savant.utils.logging import get_logger, init_logging

logger = get_logger('uri-input')

ADAPTERS_DIR = Path(__file__).parent.parent / 'adapters' / 'gst' / 'sources'


class SourceAdapter(Enum):
    FFMPEG = 'ffmpeg'
    MEDIA_FILE = 'media-file'


def validate_source_id(ctx, param, value):
    if value is None:
        return value
    safe_chars = set(string.ascii_letters + string.digits + '_.-')
    invalid_chars = {char for char in value if char not in safe_chars}
    if len(invalid_chars) > 0:
        raise click.BadParameter(f'chars {invalid_chars} are not allowed.')
    return value


@click.command()
@click.option(
    '--socket',
    default='dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc',
    help='Adapter output (module input) ZeroMQ socket endpoint.',
    show_default=True,
)
@click.option(
    '--source-id',
    default='test',
    callback=validate_source_id,
    help='Source ID, e.g. "camera1".',
    show_default=True,
)
@click.option(
    '--ffmpeg-params',
    help=(
        'A comma separated string "key=value" with parameters for FFmpeg '
        '(e.g. "rtsp_transport=tcp", "input_format=mjpeg,video_size=1280x720").'
    ),
)
@click.option(
    '--ffmpeg-buffer-len',
    help='Maximum amount of frames in the buffer for FFmpeg.',
)
@click.option(
    '--ffmpeg-loglevel',
    help='Log level for FFmpeg.',
)
@click.option(
    '--sync',
    is_flag=True,
    default=False,
    help='Send frames from source synchronously (i.e. at the source file rate).',
)
@click.option(
    '--fps-output',
    help='Where to dump stats (stdout or logger).',
)
@click.option(
    '--fps-period-frames',
    type=int,
    help='FPS measurement period, in frames.',
)
@click.option(
    '--fps-period-seconds',
    type=float,
    help='FPS measurement period, in seconds.',
)
@click.argument('uri', required=True)
def run_source(
    socket: str,
    source_id: str,
    ffmpeg_params: Optional[str],
    ffmpeg_buffer_len: Optional[int],
    ffmpeg_loglevel: Optional[str],
    sync: bool,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: Optional[str],
    uri: str,
):
    """Run source adapter with specified URI.

    \b
    URI can be:
    - path to a single file (video or picture);
    - HTTP(S) URL to a single file (video or picture);
    - path to video device (e.g. /dev/video0);
    - RTSP URL.

    Note: this script should be run inside the module container.
    """

    init_logging()
    source_adapter, uri = parse_uri(uri)

    if source_adapter == SourceAdapter.FFMPEG:
        run_ffmpeg_source(
            socket=socket,
            source_id=source_id,
            ffmpeg_params=ffmpeg_params,
            ffmpeg_buffer_len=ffmpeg_buffer_len,
            ffmpeg_loglevel=ffmpeg_loglevel,
            sync=sync,
            fps_period_frames=fps_period_frames,
            fps_period_seconds=fps_period_seconds,
            fps_output=fps_output,
            uri=uri,
        )

    elif source_adapter == SourceAdapter.MEDIA_FILE:
        run_media_file_source(
            socket=socket,
            source_id=source_id,
            sync=sync,
            fps_period_frames=fps_period_frames,
            fps_period_seconds=fps_period_seconds,
            fps_output=fps_output,
            uri=uri,
        )

    else:
        raise ValueError(f'Unsupported URI: {uri!r}')


def run_ffmpeg_source(
    socket: str,
    source_id: str,
    ffmpeg_params: Optional[str],
    ffmpeg_buffer_len: Optional[int],
    ffmpeg_loglevel: Optional[str],
    sync: bool,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: Optional[str],
    uri: str,
):
    logger.info('Running ffmpeg source with URI %r', uri)
    env = {
        'SOURCE_ID': source_id,
        'URI': uri,
        'ZMQ_ENDPOINT': socket,
        'SYNC_OUTPUT': sync,
        'FPS_OUTPUT': fps_output,
        'FPS_PERIOD_SECONDS': fps_period_seconds,
        'FPS_PERIOD_FRAMES': fps_period_frames,
        'FFMPEG_PARAMS': ffmpeg_params,
        'BUFFER_LEN': ffmpeg_buffer_len,
        'FFMPEG_LOGLEVEL': ffmpeg_loglevel,
    }
    _run_source_adapter('ffmpeg.sh', env)


def run_media_file_source(
    socket: str,
    source_id: str,
    sync: bool,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: Optional[str],
    uri: str,
):
    logger.info('Running video source with URI %r', uri)
    file_type = parse_file_type(uri)
    env = {
        'SOURCE_ID': source_id,
        'LOCATION': uri,
        'ZMQ_ENDPOINT': socket,
        'FILE_TYPE': file_type.value,
        'SYNC_OUTPUT': sync,
        'FPS_OUTPUT': fps_output,
        'FPS_PERIOD_SECONDS': fps_period_seconds,
        'FPS_PERIOD_FRAMES': fps_period_frames,
    }
    _run_source_adapter('media_files.sh', env)


def _run_source_adapter(adapter: str, env: Dict[str, Any]):
    command = [str((ADAPTERS_DIR / adapter).absolute())]
    env = {k: str(v) for k, v in env.items() if v is not None}
    env.update(os.environ)

    try:
        subprocess.check_call(command, env=env)
    except subprocess.CalledProcessError as err:
        exit(err.returncode)


def parse_uri(uri: str):
    parsed = urlparse(uri)
    if parsed.scheme == 'rtsp':
        return SourceAdapter.FFMPEG, uri

    if parsed.scheme in ['http', 'https']:
        return SourceAdapter.MEDIA_FILE, uri

    if parsed.scheme == 'file' or not parsed.scheme:
        path = parsed.path
        if path.startswith('/dev/'):
            return SourceAdapter.FFMPEG, path

        return SourceAdapter.MEDIA_FILE, path

    return None, uri


def parse_file_type(uri: str) -> FileType:
    parsed = urlparse(uri)
    if parsed.scheme in ['http', 'https']:
        mime_type = mimetypes.guess_type(parsed.path)[0]
    else:
        if not os.path.exists(uri):
            raise ValueError(f'{uri!r} does not exist')
        if not os.path.isfile(uri):
            raise ValueError(f'{uri!r} is not a file')
        mime_type = parse_mime_types([Path(uri)])[0][1]

    file_type = FileType.from_mime_type(mime_type)
    if file_type is None:
        raise ValueError(f'Failed to detect file type for {uri!r}')

    return file_type


if __name__ == '__main__':
    run_source()
