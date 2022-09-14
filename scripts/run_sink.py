#!/usr/bin/env python3
"""Run sink adapter."""
import os

import click

from common import build_docker_run_command, adapter_docker_image_option, run_command


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
        is_flag=True,
        default=False,
        help=(
            'Adapter input (module output) ZeroMQ socket bind/connect mode '
            '(bind if True).'
        ),
        show_default=True,
    )(func)
    return func


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
@adapter_docker_image_option('deepstream')
def display_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    sync: bool,
    docker_image: str,
    closing_delay: int,
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

    cmd = build_docker_run_command(
        'sink-display',
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
        sync=sync,
        entrypoint='/opt/app/adapters/ds/sinks/display.sh',
        envs=['DISPLAY', f'XAUTHORITY={xauth}', f'CLOSING_DELAY={closing_delay}'],
        volumes=[f'{xsock}:{xsock}', f'{xauth}:{xauth}'],
        with_gpu=True,
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('meta-json')
@skip_frames_without_objects_option
@chunk_size_option(0)
@common_options
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
):
    """Write metadata to streaming JSON files.

    LOCATION - location of the file to write. Can be plain location or pattern.
    (e.g. `/data/meta-%source_id-%src_filename`).
    Allowed substitution parameters: `%source_id`, `%src_filename`.
    """

    location = os.path.abspath(location)
    target_dir = os.path.dirname(location.split('%', 1)[0])

    cmd = build_docker_run_command(
        'sink-meta-json',
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
        entrypoint='/opt/app/adapters/python/sinks/metadata_json.py',
        envs=[
            f'LOCATION={location}',
            f'SKIP_FRAMES_WITHOUT_OBJECTS={skip_frames_without_objects}',
            f'CHUNK_SIZE={chunk_size}',
        ],
        volumes=[f'{target_dir}:{target_dir}'],
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('image-files')
@skip_frames_without_objects_option
@chunk_size_option()
@common_options
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
):
    """Write metadata to image files.

    LOCATION - target directory for the files to write.
    """

    location = os.path.abspath(location)
    target_dir = os.path.dirname(location.split('%', 1)[0])

    cmd = build_docker_run_command(
        'sink-image-files',
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
        entrypoint='/opt/app/adapters/python/sinks/image_files.py',
        envs=[
            f'DIR_LOCATION={location}',
            f'SKIP_FRAMES_WITHOUT_OBJECTS={skip_frames_without_objects}',
            f'CHUNK_SIZE={chunk_size}',
        ],
        volumes=[f'{target_dir}:{target_dir}'],
        docker_image=docker_image,
    )
    run_command(cmd)


@cli.command('video-files')
@chunk_size_option()
@common_options
@adapter_docker_image_option('gstreamer')
@click.argument('location', required=True)
def video_files_sink(
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    docker_image: str,
    chunk_size: int,
    location: str,
):
    """Write video to files at directory LOCATION."""

    location = os.path.abspath(location)

    cmd = build_docker_run_command(
        'sink-video-files',
        zmq_endpoint=in_endpoint,
        zmq_type=in_type,
        zmq_bind=in_bind,
        entrypoint='/opt/app/adapters/gst/sinks/video_files.sh',
        envs=[
            f'DIR_LOCATION={location}',
            f'CHUNK_SIZE={chunk_size}',
        ],
        volumes=[f'{location}:{location}'],
        docker_image=docker_image,
    )
    run_command(cmd)


if __name__ == '__main__':
    cli()
