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
        f'IN_ENDPOINT={in_endpoint}',
        f'OUT_ENDPOINT={out_endpoint}',
    ]


@cli.command('buffer')
@common_options
@click.option(
    '--queue-capacity',
    default=1000,
    help='Maximum amount of messages in the buffer.',
    show_default=True,
)
@click.option(
    '--mount-queue-path',
    default=False,
    is_flag=True,
    help='Mount queue path to the container.',
)
@click.option(
    '--interval',
    default=1.0,
    help='Interval between pushing/polling messages to the buffer, in seconds.',
    show_default=True,
)
@click.option(
    '--stats-log-interval',
    default=60,
    help='Interval between logging stats, in seconds.',
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
@click.argument('queue_path', required=True)
def buffer_bridge(
    in_endpoint: str,
    out_endpoint: str,
    queue_capacity: int,
    mount_queue_path: bool,
    interval: float,
    stats_log_interval: int,
    metrics_frame_period: int,
    metrics_time_period: Optional[float],
    metrics_history: int,
    metrics_provider: Optional[str],
    metrics_provider_params: str,
    docker_image: str,
    queue_path: str,
):
    if mount_queue_path:
        volumes = [f'{queue_path}:{queue_path}']
    else:
        volumes = []

    envs = build_common_bridge_envs(
        in_endpoint=in_endpoint,
        out_endpoint=out_endpoint,
    ) + [
        f'QUEUE_CAPACITY={queue_capacity}',
        f'QUEUE_PATH={queue_path}',
        f'INTERVAL={interval}',
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
