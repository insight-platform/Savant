#!/usr/bin/env python3
"""Run module."""
import pathlib
from typing import Optional

import click
from common import (
    detach_option,
    docker_image_option,
    get_docker_runtime,
    get_ipc_mounts,
    get_tcp_parameters,
    run_command,
)


def get_downloads_mount(
    host_parent_dir: pathlib.Path, sample_name: str, container_dir: str
):
    host_downloads_dir = (host_parent_dir / 'downloads' / sample_name).resolve()
    return f'{host_downloads_dir}:{container_dir}'


def get_models_mount(
    host_parent_dir: pathlib.Path, sample_name: str, container_dir: str
):
    host_models_dir = (host_parent_dir / 'models' / sample_name).resolve()
    return f'{host_models_dir}:{container_dir}'


@click.argument('module-config')
@click.argument('module-args', nargs=-1)
@click.option(
    '--in-endpoint',
    default='ipc:///tmp/zmq-sockets/input-video.ipc',
    help='Input ZeroMQ socket endpoint',
    show_default=True,
)
@click.option(
    '--in-type',
    default='ROUTER',
    help='Input ZeroMQ socket type',
    show_default=True,
)
@click.option(
    '--in-bind',
    default=True,
    help='Input ZeroMQ socket bind/connect mode (bind if True)',
    show_default=True,
)
@click.option(
    '--out-endpoint',
    default='ipc:///tmp/zmq-sockets/output-video.ipc',
    help='Output ZeroMQ socket endpoint',
    show_default=True,
)
@click.option(
    '--out-type',
    default='PUB',
    help='Output ZeroMQ socket type',
    show_default=True,
)
@click.option(
    '--out-bind',
    default=True,
    help='Output ZeroMQ socket bind/connect mode (bind if True)',
    show_default=True,
)
@docker_image_option('savant-deepstream')
@detach_option
def run_module(
    module_config: str,
    module_args: list,
    in_endpoint: str,
    in_type: str,
    in_bind: bool,
    out_endpoint: str,
    out_type: str,
    out_bind: bool,
    docker_image: Optional[str],
    detach: bool,
):
    """Run sample module."""
    repo_root_dir = pathlib.Path(__file__).parent.parent

    container_downloads_dir = '/downloads'
    container_model_dir = '/models'
    # fmt: off
    command = [
        'docker', 'run',
        '--rm',
        '-e', f'DOWNLOAD_PATH={container_downloads_dir}',
        '-e', f'MODEL_PATH={container_model_dir}',
        '-e', 'GST_DEBUG',
        '-e', 'LOGLEVEL',
        '-e', 'METRICS_FRAME_PERIOD',
        '-e', 'GST_DEBUG_COLOR_MODE=off',
        '-e', f'ZMQ_SRC_ENDPOINT={in_endpoint}',
        '-e', f'ZMQ_SRC_TYPE={in_type}',
        '-e', f'ZMQ_SRC_BIND={in_bind}',
        '-e', f'ZMQ_SINK_ENDPOINT={out_endpoint}',
        '-e', f'ZMQ_SINK_TYPE={out_type}',
        '-e', f'ZMQ_SINK_BIND={out_bind}',
    ]
    # fmt: on

    if detach:
        command += ['--detach']

    command += get_tcp_parameters((in_endpoint, out_endpoint))

    module_config_path = pathlib.Path(module_config)
    if module_config_path.suffix != '.yml':
        raise click.BadParameter(
            'Path to module config is expected to end with ".yml" suffix.'
        )

    # module name = module folder name, samples/<folder>
    if module_config_path.parts[0] != 'samples':
        raise click.BadParameter(
            'Currently it is only supported to run modules from the "samples/" folder.'
        )
    module_name = module_config_path.parts[1]

    volumes = [
        f'{(repo_root_dir / "samples").resolve()}:/opt/savant/samples',
        f'{(repo_root_dir / "data").resolve()}:/data:ro',
        get_downloads_mount(repo_root_dir, module_name, container_downloads_dir),
        get_models_mount(repo_root_dir, module_name, container_model_dir),
    ]
    volumes += get_ipc_mounts((in_endpoint, out_endpoint))
    for volume in volumes:
        command += ['-v', volume]

    command.append(get_docker_runtime())
    command.append(docker_image)
    # entrypoint arg
    command.append(str(module_config_path))
    if module_args:
        command.extend(module_args)

    run_command(command)


main = click.command()(run_module)

if __name__ == '__main__':
    main()
