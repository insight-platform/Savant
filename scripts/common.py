"""Common utilities for run scripts."""
from pathlib import Path
from typing import List, Iterable, Optional
import sys
import os
import pathlib
import subprocess

import click

sys.path.append(str(Path(__file__).parent.parent))
from savant.utils.version import version  # noqa: F401
from savant.utils.platform import is_aarch64, get_l4t_version  # noqa: F401

# docker registry to use with scripts, set to "None" to use local images
# DOCKER_REGISTRY = 'ghcr.io/insight-platform'
DOCKER_REGISTRY = None


def docker_image_option(default_docker_image_name: str, tag: Optional[str] = None):
    """Click option for docker image."""
    SAVANT_VERSION = version.SAVANT
    DEEPSTREAM_VERSION = version.DEEPSTREAM
    if is_aarch64() and get_l4t_version()[0] == 32:
        DEEPSTREAM_VERSION = '6.0.1'

    if is_aarch64():
        default_docker_image_name += '-l4t'

    default_tag = SAVANT_VERSION
    if 'deepstream' in default_docker_image_name:
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
        transport, address = zmq_socket.split('://')
        if transport == 'ipc':
            ipc_mounts.append(get_ipc_mount(address))

    return list(set(ipc_mounts))


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

    command.append(docker_image)

    return command


def run_command(command: List[str]):
    """Start a subprocess, call command."""
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as err:
        sys.exit(err.returncode)
