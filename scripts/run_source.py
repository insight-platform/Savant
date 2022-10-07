#!/usr/bin/env python3
"""Run source adapter."""
import os
from typing import List, Optional
import string

import click

from common import build_docker_run_command, adapter_docker_image_option, run_command


@click.group()
def cli():
    """Click command line group callback."""


sync_option = click.option(
    '--sync',
    is_flag=True,
    default=False,
    help='Send frames from source synchronously (i.e. at the source file rate).',
    show_default=True,
)


def validate_source_id(ctx, param, value):
    safe_chars = set(string.ascii_letters + string.digits + '_.-')
    invalid_chars = {char for char in value if char not in safe_chars}
    if len(invalid_chars) > 0:
        raise click.BadParameter(f'chars {invalid_chars} are not allowed.')
    return value


def common_options(func):
    """Common Click source adapter options."""
    func = click.option(
        '--out-endpoint',
        default='ipc:///tmp/zmq-sockets/input-video.ipc',
        help='Adapter output (module input) ZeroMQ socket endpoint.',
        show_default=True,
    )(func)
    func = click.option(
        '--out-type',
        default='REQ',
        help='Adapter output (module input) ZeroMQ socket type.',
        show_default=True,
    )(func)
    func = click.option(
        '--out-bind',
        default=False,
        help=(
            'Adapter output (module input) ZeroMQ socket bind/connect mode '
            '(bind if True).'
        ),
        show_default=True,
    )(func)
    func = click.option(
        '--fps-output',
        help='Where to dump stats (stdout or logger).',
    )(func)
    func = click.option(
        '--fps-period-frames',
        type=int,
        help='FPS measurement period, in frames.',
    )(func)
    func = click.option(
        '--fps-period-seconds',
        type=float,
        help='FPS measurement period, in seconds.',
    )(func)
    func = click.option(
        '--source-id',
        required=True,
        type=click.STRING,
        callback=validate_source_id,
        help='Source ID, e.g. "camera1".',
    )(func)
    return func


def build_common_envs(
    source_id: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
):
    """Generate env var run options."""
    envs = [f'SOURCE_ID={source_id}']
    if fps_period_frames:
        envs.append(f'FPS_PERIOD_FRAMES={fps_period_frames}')
    if fps_period_seconds:
        envs.append(f'FPS_PERIOD_SECONDS={fps_period_seconds}')
    if fps_output:
        envs.append(f'FPS_OUTPUT={fps_output}')
    return envs


def files_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    location: str,
    file_type: str,
    envs: List[str],
):
    """Read picture or video files from LOCATION.
    LOCATION can be single file, directory or HTTP URL.
    """
    print(source_id)
    if location.startswith('http://') or location.startswith('https://'):
        volumes = []
    else:
        assert os.path.exists(location)
        location = os.path.abspath(location)
        volumes = [f'{location}:{location}:ro']

    cmd = build_docker_run_command(
        f'source-{file_type}-files-{source_id}',
        zmq_endpoint=out_endpoint,
        zmq_type=out_type,
        zmq_bind=out_bind,
        sync=sync,
        entrypoint='/opt/app/adapters/gst/sources/media_files.sh',
        envs=(
            build_common_envs(
                source_id=source_id,
                fps_period_frames=fps_period_frames,
                fps_period_seconds=fps_period_seconds,
                fps_output=fps_output,
            )
            + [f'LOCATION={location}', f'FILE_TYPE={file_type}']
            + envs
        ),
        volumes=volumes,
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('videos')
@click.option(
    '--sort-by-time',
    default=False,
    is_flag=True,
    help='Sort files by modification time.',
    show_default=True,
)
@common_options
@sync_option
@adapter_docker_image_option('gstreamer')
@click.argument('location', required=True)
def videos_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    location: str,
    sort_by_time: bool,
):
    """Read video files from LOCATION.
    LOCATION can be single file, directory or HTTP URL.
    """

    files_source(
        source_id=source_id,
        out_endpoint=out_endpoint,
        out_type=out_type,
        out_bind=out_bind,
        sync=sync,
        docker_image=docker_image,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        location=location,
        file_type='video',
        envs=[f'SORT_BY_TIME={sort_by_time}'],
    )


@cli.command('pictures')
@click.option(
    '--framerate',
    default='30/1',
    help='Frame rate of the pictures.',
    show_default=True,
)
@click.option(
    '--sort-by-time',
    default=False,
    is_flag=True,
    help='Sort files by modification time.',
    show_default=True,
)
@common_options
@sync_option
@adapter_docker_image_option('gstreamer')
@click.argument('location', required=True)
def pictures_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    sync: bool,
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    location: str,
    framerate: str,
    sort_by_time: bool,
):
    """Read picture files from LOCATION.
    LOCATION can be single file, directory or HTTP URL.
    """

    files_source(
        source_id=source_id,
        out_endpoint=out_endpoint,
        out_type=out_type,
        out_bind=out_bind,
        sync=sync,
        docker_image=docker_image,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
        location=location,
        file_type='picture',
        envs=[
            f'FRAMERATE={framerate}',
            f'SORT_BY_TIME={sort_by_time}',
        ],
    )


@cli.command('rtsp')
@common_options
@adapter_docker_image_option('gstreamer')
@click.argument('rtsp_uri', required=True)
def rtsp_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    rtsp_uri: str,
):
    """Read video stream from RTSP_URI."""

    cmd = build_docker_run_command(
        f'source-rtsp-{source_id}',
        zmq_endpoint=out_endpoint,
        zmq_type=out_type,
        zmq_bind=out_bind,
        entrypoint='/opt/app/adapters/gst/sources/rtsp.sh',
        envs=(
            build_common_envs(
                source_id=source_id,
                fps_period_frames=fps_period_frames,
                fps_period_seconds=fps_period_seconds,
                fps_output=fps_output,
            )
            + [f'RTSP_URI={rtsp_uri}']
        ),
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('usb-cam')
@click.option(
    '--framerate',
    default='15/1',
    help='USB camera framerate',
    show_default=True,
)
@common_options
@adapter_docker_image_option('gstreamer')
@click.argument('device', default='/dev/video0')
def usb_cam_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    framerate: str,
    device: str,
):
    """Read video stream from USB camera located at DEVICE.

    Default DEVICE: /dev/video0.
    """

    cmd = build_docker_run_command(
        f'source-usb-{source_id}',
        zmq_endpoint=out_endpoint,
        zmq_type=out_type,
        zmq_bind=out_bind,
        entrypoint='/opt/app/adapters/gst/sources/usb_cam.sh',
        envs=(
            build_common_envs(
                source_id=source_id,
                fps_period_frames=fps_period_frames,
                fps_period_seconds=fps_period_seconds,
                fps_output=fps_output,
            )
            + [f'DEVICE={device}', f'FRAMERATE={framerate}']
        ),
        devices=[device],
        docker_image=docker_image,
    )
    run_command(cmd)


if __name__ == '__main__':
    cli()
