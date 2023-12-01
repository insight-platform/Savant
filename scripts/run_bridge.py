#!/usr/bin/env python3
"""Run bridge adapter."""
import uuid
from typing import Optional

import click
from common import adapter_docker_image_option, build_docker_run_command, run_command


@click.group()
def cli():
    """Click command line group callback."""


def common_options(func):
    """Common Click bridge adapter options."""
    func = click.option(
        '--in-endpoint',
        help='Adapter input ZeroMQ socket endpoint.',
        required=True,
    )(func)
    func = click.option(
        '--out-endpoint',
        help='Adapter output ZeroMQ socket endpoint.',
        required=True,
    )(func)
    return func


def build_common_bridge_envs(
    in_endpoint: str,
    out_endpoint: str,
):
    """Generate env var run options."""
    return [
        f'ZMQ_SRC_ENDPOINT={in_endpoint}',
        f'ZMQ_SINK_ENDPOINT={out_endpoint}',
    ]


@cli.command('buffer')
@common_options
@click.option(
    '--buffer-len',
    default=1000,
    help='Maximum amount of messages in the buffer.',
    show_default=True,
)
@click.option(
    '--mount-buffer-path',
    default=False,
    is_flag=True,
    help='Mount buffer path to the container.',
)
@click.option(
    '--buffer-service-messages',
    default=100,
    help=(
        'Buffer length for service messages (eg. EndOfStream, Shutdown). Used '
        'when the main part of the buffer is full (--buffer-len).'
    ),
    show_default=True,
)
@click.option(
    '--buffer-threshold-percentage',
    default=80,
    help='Threshold to mark the buffer not full.',
    show_default=True,
)
@click.option(
    '--idle-polling-period',
    default=0.005,
    help=(
        'Interval between polling messages from the buffer '
        'when the buffer is empty, in seconds.'
    ),
    show_default=True,
)
@click.option(
    '--stats-log-interval',
    default=60,
    help='Interval between logging buffer statistics, in seconds.',
    show_default=True,
)
@click.option(
    '--metrics-frame-period',
    default=1000,
    help='Output FPS stats after every N frames.',
    show_default=True,
)
@click.option(
    '--metrics-time-period',
    type=float,
    help='Output FPS stats after every N seconds.',
)
@click.option(
    '--metrics-history',
    default=100,
    help='How many last FPS stats to keep in the memory.',
    show_default=True,
)
@click.option(
    '--metrics-provider',
    help='Metrics provider name.',
)
@click.option(
    '--metrics-provider-params',
    default='{}',
    help='JSON dict of metrics provider parameters.',
    show_default=True,
)
@adapter_docker_image_option('py')
@click.argument('buffer_path', required=True)
def buffer_bridge(
    in_endpoint: str,
    out_endpoint: str,
    buffer_len: int,
    mount_buffer_path: bool,
    buffer_service_messages: int,
    buffer_threshold_percentage: int,
    idle_polling_period: float,
    stats_log_interval: int,
    metrics_frame_period: int,
    metrics_time_period: Optional[float],
    metrics_history: int,
    metrics_provider: Optional[str],
    metrics_provider_params: str,
    docker_image: str,
    buffer_path: str,
):
    """Buffers messages from a source to BUFFER_PATH and sends them to a module.

    When the module is not able to accept the message, the adapter buffers it
    until the module is ready to accept it. When the buffer is full, the adapter
    drops the incoming message.
    """

    if mount_buffer_path:
        volumes = [f'{buffer_path}:{buffer_path}']
    else:
        volumes = []

    envs = build_common_bridge_envs(
        in_endpoint=in_endpoint,
        out_endpoint=out_endpoint,
    ) + [
        f'BUFFER_LEN={buffer_len}',
        f'BUFFER_PATH={buffer_path}',
        f'BUFFER_SERVICE_MESSAGES={buffer_service_messages}',
        f'BUFFER_THRESHOLD_PERCENTAGE={buffer_threshold_percentage}',
        f'IDLE_POLLING_PERIOD={idle_polling_period}',
        f'STATS_LOG_INTERVAL={stats_log_interval}',
        f'METRICS_FRAME_PERIOD={metrics_frame_period}',
        f'METRICS_HISTORY={metrics_history}',
        f'METRICS_PROVIDER_PARAMS={metrics_provider_params}',
    ]
    if metrics_provider:
        envs.append(f'METRICS_PROVIDER={metrics_provider}')
    if metrics_time_period:
        envs.append(f'METRICS_TIME_PERIOD={metrics_time_period}')

    cmd = build_docker_run_command(
        container_name=f'buffer-bridge-{uuid.uuid4().hex}',
        zmq_endpoints=[in_endpoint, out_endpoint],
        entrypoint='python',
        args=['-m', 'adapters.python.bridge.buffer'],
        docker_image=docker_image,
        envs=envs,
        volumes=volumes,
    )
    run_command(cmd)


if __name__ == '__main__':
    cli()
