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


@cli.command('gige')
@click.option('--width', type=int, help='Width of streaming video')
@click.option('--height', type=int, help='Height of streaming video')
@click.option('--framerate', type=str, help='Framerate of streaming video')
@click.option(
    # TODO: replace with PixelFormat
    # https://github.com/AravisProject/aravis/blob/0.8.22/src/arvmisc.c#L656
    '--input-caps',
    type=str,
    help='Caps of input video (e.g. "video/x-raw,format=RGB"). Look '
    'https://github.com/AravisProject/aravis/blob/0.8.22/src/arvmisc.c#L656 '
    'for PixelFormat -> Caps mapping.',
)
@click.option('--packet-size', type=int, help='GigEVision streaming packet size')
@click.option(
    '--auto-packet-size', type=bool, help='Negotiate GigEVision streaming packet size'
)
@click.option('--exposure', type=float, help='Exposure time (µs)')
@click.option('--exposure-auto', help='Auto Exposure Mode, one of "off", "once", "on"')
@click.option('--gain', type=float, help='Gain (dB)')
@click.option('--gain-auto', help='Auto Gain Mode, one of "off", "once", "on"')
@click.option(
    '--features',
    help='Additional configuration parameters as a space separated list of feature assignations',
)
@click.option(
    '--host-network', default=False, is_flag=True, help='Use the host network'
)
@common_options
@adapter_docker_image_option('gstreamer')
@click.argument('camera_name', required=False)
def gige_cam_source(
    source_id: str,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    docker_image: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
    width: Optional[int],
    height: Optional[int],
    framerate: Optional[str],
    input_caps: Optional[str],
    packet_size: Optional[int],
    auto_packet_size: Optional[bool],
    exposure: Optional[float],
    exposure_auto: Optional[str],
    gain: Optional[float],
    gain_auto: Optional[str],
    features: Optional[str],
    host_network: bool,
    camera_name: Optional[str],
):
    """Read video stream from GigE camera CAMERA_NAME.

    If the camera is a GigEVision, CAMERA_NAME can be either:

      - <vendor>-<model>-<serial>

      - <vendor_alias>-<serial>

      - <vendor>-<serial>

      - <user_id>

      - <ip_address>

      - <mac_address>
    """

    envs = build_common_envs(
        source_id=source_id,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
    )

    if camera_name is not None:
        envs.append(f'CAMERA_NAME={camera_name}')
    if width is not None:
        envs.append(f'WIDTH={width}')
    if height is not None:
        envs.append(f'HEIGHT={height}')
    if framerate is not None:
        envs.append(f'FRAMERATE={framerate}')
    if input_caps is not None:
        envs.append(f'INPUT_CAPS={input_caps}')
    if packet_size is not None:
        envs.append(f'PACKET_SIZE={packet_size}')
    if auto_packet_size is not None:
        envs.append(f'AUTO_PACKET_SIZE={int(auto_packet_size)}')
    if exposure is not None:
        envs.append(f'EXPOSURE={exposure}')
    if exposure_auto is not None:
        envs.append(f'EXPOSURE_AUTO={exposure_auto}')
    if gain is not None:
        envs.append(f'GAIN={gain}')
    if gain_auto is not None:
        envs.append(f'GAIN_AUTO={gain_auto}')
    if features is not None:
        envs.append(f'FEATURES={features}')

    cmd = build_docker_run_command(
        f'source-gige-{source_id}',
        zmq_endpoint=out_endpoint,
        zmq_type=out_type,
        zmq_bind=out_bind,
        entrypoint='/opt/app/adapters/gst/sources/gige_cam.sh',
        envs=envs,
        docker_image=docker_image,
        host_network=host_network,
    )
    run_command(cmd)


if __name__ == '__main__':
    cli()
