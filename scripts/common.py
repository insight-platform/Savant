"""Common utilities for run scripts."""
import os
import pathlib
import string
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import click

sys.path.append(str(Path(__file__).parent.parent))
from savant.utils.platform import is_aarch64
from savant.utils.re_patterns import socket_uri_pattern
from savant.utils.version import version

SAVANT_VERSION = 'latest'  # use version.SAVANT or 'latest'
DEEPSTREAM_VERSION = version.DEEPSTREAM

# docker registry to use with scripts, set to "None" to use local images
DOCKER_REGISTRY = 'ghcr.io/insight-platform'
# DOCKER_REGISTRY = None


def docker_image_option(default_docker_image_name: str, tag: Optional[str] = None):
    """Click option for docker image."""
    if is_aarch64():
        default_docker_image_name += '-l4t'

    default_tag = SAVANT_VERSION
    if 'deepstream' in default_docker_image_name and SAVANT_VERSION != 'latest':
        default_tag += f'-{DEEPSTREAM_VERSION}'

    if tag:
        default_tag += f'-{tag}'

    registry = DOCKER_REGISTRY.strip('/') + '/' if DOCKER_REGISTRY else ''
    default_docker_image = f'{registry}{default_docker_image_name}:{default_tag}'
    return click.option(
        '-i',
        '--docker-image',
        default=default_docker_image,
        help=f'Name of docker image.',
        show_default=True,
    )


def adapter_docker_image_option(default_suffix: str):
    return docker_image_option(f'savant-adapters-{default_suffix}')


def get_tcp_parameters(zmq_sockets: Iterable[str]) -> List[str]:
    """Get necessary docker run parameters for tcp zmq socket."""
    for zmq_socket in zmq_sockets:
        transport, _ = zmq_socket.split('://')
        if transport == 'tcp':
            return ['--network', 'host']
    return []


def get_ipc_mount(address: str) -> str:
    """Get mount dir for a single endpoint address."""
    zmq_socket_dir = pathlib.Path(address).parent
    return f'{zmq_socket_dir}:{zmq_socket_dir}'


def get_ipc_mounts(zmq_sockets: Iterable[str]) -> List[str]:
    """Get mount dirs for zmq sockets."""
    ipc_mounts = []

    for zmq_socket in zmq_sockets:
        _, endpoint = socket_uri_pattern.fullmatch(zmq_socket).groups()
        transport, address = endpoint.split('://')
        if transport == 'ipc':
            ipc_mounts.append(get_ipc_mount(address))

    return list(set(ipc_mounts))


def validate_source_id(ctx, param, value):
    if value is None:
        return value
    safe_chars = set(string.ascii_letters + string.digits + '_.-')
    invalid_chars = {char for char in value if char not in safe_chars}
    if len(invalid_chars) > 0:
        raise click.BadParameter(f'chars {invalid_chars} are not allowed.')
    return value


def validate_source_id_list(ctx, param, value):
    if value is None:
        return value
    for source_id in value.split(','):
        validate_source_id(ctx, param, source_id)
    return value


def source_id_option(required: bool):
    return click.option(
        '--source-id',
        required=required,
        type=click.STRING,
        callback=validate_source_id,
        help='Source ID, e.g. "camera1".',
    )


def fps_meter_options(func):
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
    return func


def build_common_envs(
    source_id: Optional[str],
    fps_period_frames: Optional[int],
    fps_period_seconds: Optional[float],
    fps_output: str,
):
    """Generate env var run options."""
    envs = []
    if source_id:
        envs.append(f'SOURCE_ID={source_id}')
    if fps_period_frames:
        envs.append(f'FPS_PERIOD_FRAMES={fps_period_frames}')
    if fps_period_seconds:
        envs.append(f'FPS_PERIOD_SECONDS={fps_period_seconds}')
    if fps_output:
        envs.append(f'FPS_OUTPUT={fps_output}')
    return envs


def build_docker_run_command(
    container_name: str,
    zmq_endpoint: str,
    zmq_type: str,
    zmq_bind: bool,
    entrypoint: str,
    docker_image: str,
    sync: bool = False,
    envs: List[str] = None,
    volumes: List[str] = None,
    devices: List[str] = None,
    with_gpu: bool = False,
    host_network: bool = False,
    args: List[str] = None,
    ports: List[Tuple[int, int]] = None,
) -> List[str]:
    """Build docker run command for an adapter container.

    :param container_name: run container with this name
    :param zmq_endpoint: add ``ZMQ_ENDPOINT`` env var to container
        that will specify zmq socket endpoint, eg.
        ``ipc:///tmp/zmq-sockets/input-video.ipc``  or
        ``tcp://0.0.0.0:5000``
    :param zmq_type: add ``ZMQ_TYPE`` env var to container
    :param zmq_bind: add ``ZMQ_BIND`` env var to container
    :param entrypoint: add ``--entrypoint`` parameter
    :param docker_image: docker image to run
    :param sync: add ``SYNC_OUTPUT`` env var to container
    :param envs: add ``-e`` parameters
    :param volumes: add ``-v`` parametrs
    :param devices: add ``--devices`` parameters
    :param with_gpu: add ``--gpus=all`` parameter
    :param host_network: add ``--network=host`` parameter
    :param args: add command line arguments to the entrypoint
    :param ports: add ``-p`` parameters
    """
    gst_debug = os.environ.get('GST_DEBUG', '2')
    # fmt: off
    command = [
        'docker', 'run',
        '--rm',
        '-it',
        '--name', container_name,
        '-e', f'GST_DEBUG={gst_debug}',
        '-e', 'LOGLEVEL',
        '-e', f'SYNC_OUTPUT={sync}',
        '-e', f'ZMQ_ENDPOINT={zmq_endpoint}',
        '-e', f'ZMQ_TYPE={zmq_type}',
        '-e', f'ZMQ_BIND={zmq_bind}',
    ]
    # fmt: on

    command += get_tcp_parameters((zmq_endpoint,))

    command += ['--entrypoint', entrypoint]

    if envs:
        for env in envs:
            command += ['-e', env]

    if volumes is None:
        volumes = []
    volumes += get_ipc_mounts((zmq_endpoint,))
    for volume in volumes:
        command += ['-v', volume]

    if devices:
        for device in devices:
            command += ['--device', device]

    if with_gpu:
        command.append('--gpus=all')

    if host_network:
        command.append('--network=host')

    if ports:
        for host_port, container_port in ports:
            command += ['-p', f'{host_port}:{container_port}']

    command.append(docker_image)
    if args:
        command.extend(args)

    return command


def run_command(command: List[str]):
    """Start a subprocess, call command."""
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as err:
        sys.exit(err.returncode)
