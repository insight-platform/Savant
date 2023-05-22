#!/usr/bin/env python3
"""Run sink adapter."""
import os
from typing import Optional
import uuid

import click

from common import (
    adapter_docker_image_option,
    build_common_envs,
    build_docker_run_command,
    fps_meter_options,
    run_command,
    source_id_option,
    validate_source_id,
)


@click.group()
def cli():
    """Click command line group callback."""


def common_options(func):
    """Common Click sink adapter options."""
    func = click.option(
        '--in-endpoint',
        default='ipc:///tmp/zmq-sockets/output-video.ipc',
        help='Adapter input (module output) ZeroMQ socket endpoint.',
        show_default=True,
    )(func)
    func = click.option(
        '--in-type',
        default='SUB',
        help='Adapter input (module output) ZeroMQ socket type.',
        show_default=True,
    )(func)
    func = click.option(
        '--in-bind',
        default=False,
        help=(
            'Adapter input (module output) ZeroMQ socket bind/connect mode '
            '(bind if True).'
        ),
        show_default=True,
    )(func)
    return func


def source_id_prefix_option(func):
    return click.option(
        '--source-id-prefix',
        callback=validate_source_id,
        help='Filter frames by source ID prefix.',
    )(func)


def build_common_sink_envs(
    source_id: Optional[str],
    source_id_prefix: Optional[str],
):
    """Generate env var run options."""
    envs = []
    if source_id:
        envs.append(f'SOURCE_ID={source_id}')
    if source_id_prefix:
        envs.append(f'SOURCE_ID_PREFIX={source_id_prefix}')
    return envs


skip_frames_without_objects_option = click.option(
    '--skip-frames-without-objects',
    is_flag=True,
    default=False,
    help='Skip frames without detected objects.',
    show_default=True,
)


def chunk_size_option(default=10000):
    """Chunk size option."""
    return click.option(
        '--chunk-size',
        default=default,
        help='Chunk size in frames.',
        show_default=True,
    )


@cli.command('display')
@click.option(
    '--closing-delay',
    type=click.INT,
    default=0,
    help='Delay in seconds before closing window when stream from source finished.',
    show_default=True,
)
@click.option(
    '--sync',
    is_flag=True,
    default=False,
    help='Show frames on sink synchronously (i.e. at the source file rate).',
    show_default=True,
)
@common_options
@source_id_option(required=False)
@source_id_prefix_option
@adapter_docker_image_option('deepstream')
def display_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    sync: bool,
    docker_image: str,
    closing_delay: int,
    source_id: Optional[str],
    source_id_prefix: Optional[str],
):
    """Show video on display, one window per source."""

    print('Setting environment variables for X')
    run_command(['xhost', '+local:docker'])
    xsock = '/tmp/.X11-unix'
    xauth = '/tmp/.docker.xauth'
    run_command(
        [
            'bash',
            '-c',
            (
                f"xauth nlist $DISPLAY | "
                "sed -e 's/^..../ffff/' | "
                f"xauth -f {xauth} nmerge -"
            ),
        ]
    )

    envs = build_common_sink_envs(
        source_id=source_id,
        source_id_prefix=source_id_prefix,
    ) + [
        'DISPLAY',
        f'XAUTHORITY={xauth}',
        f'CLOSING_DELAY={closing_delay}',
    ]

    cmd = build_docker_run_command(
        'sink-display',
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
        sync=sync,
        entrypoint='/opt/savant/adapters/ds/sinks/display.sh',
        envs=envs,
        volumes=[f'{xsock}:{xsock}', f'{xauth}:{xauth}'],
        with_gpu=True,
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('meta-json')
@skip_frames_without_objects_option
@chunk_size_option(0)
@common_options
@source_id_option(required=False)
@source_id_prefix_option
@adapter_docker_image_option('py')
@click.argument('location', required=True)
def meta_json_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    docker_image: str,
    skip_frames_without_objects: bool,
    chunk_size: int,
    location: str,
    source_id: Optional[str],
    source_id_prefix: Optional[str],
):
    """Write metadata to streaming JSON files.

    LOCATION - location of the file to write. Can be plain location or pattern.
    (e.g. `/data/meta-%source_id-%src_filename`).
    Allowed substitution parameters: `%source_id`, `%src_filename`.
    """

    location = os.path.abspath(location)
    target_dir = os.path.dirname(location.split('%', 1)[0])

    envs = build_common_sink_envs(
        source_id=source_id,
        source_id_prefix=source_id_prefix,
    ) + [
        f'LOCATION={location}',
        f'SKIP_FRAMES_WITHOUT_OBJECTS={skip_frames_without_objects}',
        f'CHUNK_SIZE={chunk_size}',
    ]

    cmd = build_docker_run_command(
        f'sink-meta-json-{uuid.uuid4().hex}',
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
        entrypoint='/opt/savant/adapters/python/sinks/metadata_json.py',
        envs=envs,
        volumes=[f'{target_dir}:{target_dir}'],
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('image-files')
@skip_frames_without_objects_option
@chunk_size_option()
@common_options
@source_id_option(required=False)
@source_id_prefix_option
@adapter_docker_image_option('py')
@click.argument('location', required=True)
def image_files_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    docker_image: str,
    skip_frames_without_objects: bool,
    chunk_size: int,
    location: str,
    source_id: Optional[str],
    source_id_prefix: Optional[str],
):
    """Write metadata to image files.

    LOCATION - target directory for the files to write.
    """

    location = os.path.abspath(location)
    target_dir = os.path.dirname(location.split('%', 1)[0])

    envs = build_common_sink_envs(
        source_id=source_id,
        source_id_prefix=source_id_prefix,
    ) + [
        f'DIR_LOCATION={location}',
        f'SKIP_FRAMES_WITHOUT_OBJECTS={skip_frames_without_objects}',
        f'CHUNK_SIZE={chunk_size}',
    ]

    cmd = build_docker_run_command(
        f'sink-image-files-{uuid.uuid4().hex}',
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
        entrypoint='/opt/savant/adapters/python/sinks/image_files.py',
        envs=envs,
        volumes=[f'{target_dir}:{target_dir}'],
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('video-files')
@chunk_size_option()
@common_options
@source_id_option(required=False)
@source_id_prefix_option
@adapter_docker_image_option('gstreamer')
@click.argument('location', required=True)
def video_files_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    docker_image: str,
    chunk_size: int,
    location: str,
    source_id: Optional[str],
    source_id_prefix: Optional[str],
):
    """Write video to files at directory LOCATION."""

    location = os.path.abspath(location)

    envs = build_common_sink_envs(
        source_id=source_id,
        source_id_prefix=source_id_prefix,
    ) + [
        f'DIR_LOCATION={location}',
        f'CHUNK_SIZE={chunk_size}',
    ]

    cmd = build_docker_run_command(
        f'sink-video-files-{uuid.uuid4().hex}',
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
        entrypoint='/opt/savant/adapters/gst/sinks/video_files.sh',
        envs=envs,
        volumes=[f'{location}:{location}'],
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('always-on-rtsp')
@source_id_option(required=True)
@click.option(
    '--stub-file-location',
    required=True,
    help='Location of the stub image file. Image file must be in JPEG format.',
)
@click.option(
    '--max-delay-ms',
    type=click.INT,
    default=1000,
    help='Maximum delay for the last frame in milliseconds.',
    show_default=True,
)
@click.option(
    '--transfer-mode',
    default='scale-to-fit',
    help='Transfer mode. One of: "scale-to-fit", "crop-to-fit".',
    show_default=True,
)
@click.option(
    '--protocols',
    default='tcp',
    help='Allowed lower transport protocols, e.g. "tcp+udp-mcast+udp".',
    show_default=True,
)
@click.option(
    '--latency-ms',
    type=click.INT,
    default=100,
    help='Amount of ms to buffer RTSP stream.',
    show_default=True,
)
@click.option(
    '--keep-alive',
    type=click.BOOL,
    default=True,
    help='Send RTSP keep alive packets, disable for old incompatible server.',
    show_default=True,
)
@click.option(
    '--profile',
    default='High',
    help='H264 encoding profile. One of: "Baseline", "Main", "High".',
    show_default=True,
)
@click.option(
    '--bitrate',
    type=click.INT,
    default=4000000,
    help='H264 encoding bitrate.',
    show_default=True,
)
@click.option(
    '--framerate',
    default='30/1',
    help='Frame rate of the output stream.',
    show_default=True,
)
@click.option(
    '--metadata-output',
    help='Where to dump metadata (stdout or logger).',
)
@click.option(
    '--sync',
    is_flag=True,
    default=False,
    help=(
        'Show frames on sink synchronously (i.e. at the source file rate). '
        'Note: inbound stream is not stable with this flag, try to avoid it.'
    ),
    show_default=True,
)
@click.option(
    '--dev-mode',
    default=False,
    is_flag=True,
    help='Use embedded MediaMTX to publish RTSP stream.',
    show_default=True,
)
@click.option(
    '--publish-ports',
    default=False,
    is_flag=True,
    help=(
        'Publish container ports for embedded MediaMTX to the host. '
        'Published ports: 554 (RTSP), 1935 (RTMP), 888 (HLS), 8889 (WebRTC). '
        'Ignored when --dev-mode is not set.'
    ),
    show_default=True,
)
@fps_meter_options
@common_options
@adapter_docker_image_option('deepstream')
@click.argument('rtsp_uri', required=False)
def always_on_rtsp_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    docker_image: str,
    source_id: str,
    stub_file_location: str,
    max_delay_ms: int,
    transfer_mode: str,
    protocols: str,
    latency_ms: int,
    keep_alive: bool,
    profile: str,
    bitrate: int,
    framerate: str,
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: Optional[str],
    metadata_output: Optional[str],
    sync: bool,
    dev_mode: bool,
    publish_ports: bool,
    rtsp_uri: Optional[str],
):
    """Send video stream from specific source to RTSP server.

    RTSP_URI - URI of the RTSP server.
    Exactly one of --dev-mode flag and RTSP_URI argument must be used.

    When --dev-mode flag is used the stream is available at:

        - RTSP: rtsp://<container-host>:554/stream

        - RTMP: rtmp://<container-host>:1935/stream

        - HLS: http://<container-host>:888/stream

        - WebRTC: http://<container-host>:8889/stream


    Note: it is advisable to use --sync flag on source adapter or use a live
    source adapter (e.g. rtsp or usb-cam).
    """

    assert os.path.exists(stub_file_location)
    assert dev_mode == (
        rtsp_uri is None
    ), 'Must be specified one of "--dev-mode" flag or "RTSP_URI" argument.'
    stub_file_location = os.path.abspath(stub_file_location)

    envs = build_common_envs(
        source_id=source_id,
        fps_period_frames=fps_period_frames,
        fps_period_seconds=fps_period_seconds,
        fps_output=fps_output,
    ) + [
        f'STUB_FILE_LOCATION={stub_file_location}',
        f'MAX_DELAY_MS={max_delay_ms}',
        f'TRANSFER_MODE={transfer_mode}',
        f'RTSP_PROTOCOLS={protocols}',
        f'RTSP_LATENCY_MS={latency_ms}',
        f'RTSP_KEEP_ALIVE={keep_alive}',
        f'ENCODER_PROFILE={profile}',
        f'ENCODER_BITRATE={bitrate}',
        f'FRAMERATE={framerate}',
    ]
    if metadata_output:
        envs.append(f'METADATA_OUTPUT={metadata_output}')
    if dev_mode:
        envs.append(f'DEV_MODE={dev_mode}')
    else:
        envs.append(f'RTSP_URI={rtsp_uri}')

    if publish_ports:
        ports = [(x, x) for x in [554, 1935, 888, 8889]]
    else:
        ports = None

    cmd = build_docker_run_command(
        f'sink-always-on-rtsp-{uuid.uuid4().hex}',
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
        sync=sync,
        entrypoint='python',
        args=['-m', 'adapters.ds.sinks.always_on_rtsp'],
        envs=envs,
        volumes=[f'{stub_file_location}:{stub_file_location}:ro'],
        with_gpu=True,
        docker_image=docker_image,
        ports=ports,
    )
    run_command(cmd)


if __name__ == '__main__':
    cli()
