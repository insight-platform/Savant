import asyncio
import logging
import os
import signal
import sys
import time
from typing import Callable, Dict, List, Union

import aiodocker
from aiodocker import DockerError
from aiodocker.containers import DockerContainer

from .buffer_metrics import get_metrics, parse_metrics
from .config.parser import Config, ConfigParser
from .config.schema import Action, FlowConfig, QueueConfig, WatchConfig
from .config.validator import validate

LOG_LEVEL = os.environ.get('LOGLEVEL', 'INFO')

BUFFER_SIZE_METRIC = 'buffer_size'
LAST_SENT_MESSAGE_METRIC = 'last_sent_message'
LAST_RECEIVED_MESSAGE_METRIC = 'last_received_message'


def init_logging(loglevel: str):
    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=loglevel,
    )


init_logging(LOG_LEVEL)
logger = logging.getLogger('PipelineWatchdog')


class DockerClient:
    def __init__(self):
        self._client = aiodocker.Docker()

    async def get_containers(
        self, container_labels: List[List[str]]
    ) -> List[DockerContainer]:
        containers = []
        for labels in container_labels:
            try:
                containers += await self._client.containers.list(
                    all=True, filters={'label': labels}
                )
            except DockerError:
                raise RuntimeError(f'Failed to list containers with labels {labels}')

        return containers

    @staticmethod
    async def restart_container(container: DockerContainer):
        try:
            await container.restart()
            logger.debug('Container %s restarted', container.id)
        except DockerError:
            logger.error('Failed to restart container %s. Skipping', container.id)

    @staticmethod
    async def stop_container(container: DockerContainer):
        try:
            await container.stop()
            logger.debug('Container %s stopped', container.id)
        except DockerError:
            logger.error('Failed to stop container %s. Skipping', container.id)

    async def close(self):
        await self._client.close()


async def process_action(
    docker_client: DockerClient, action: Action, container_labels: List[List[str]]
):
    containers = await docker_client.get_containers(container_labels)

    if not containers:
        logger.debug('No containers found with labels %s', container_labels)
        return

    if action == Action.STOP:
        logger.debug('Stopping containers')
        for container in containers:
            await docker_client.stop_container(container)
    elif action == Action.RESTART:
        logger.debug('Restarting containers')
        for container in containers:
            await docker_client.restart_container(container)


def _queue_evaluate(
    buffer: str, config: QueueConfig, metrics: Dict[str, float]
) -> bool:
    buffer_size = metrics[BUFFER_SIZE_METRIC]
    if buffer_size > config.length:
        logger.info(
            'Queue watch [%s]: %s=%.0f exceeds threshold %s,'
            ' executing action=%s',
            buffer,
            BUFFER_SIZE_METRIC,
            buffer_size,
            config.length,
            config.action.value,
        )
        return True
    logger.info(
        'Queue watch [%s]: %s=%.0f within threshold %s, no action',
        buffer,
        BUFFER_SIZE_METRIC,
        buffer_size,
        config.length,
    )
    return False


def _flow_evaluate(
    watch_name: str,
    metric_name: str,
    buffer: str,
    config: FlowConfig,
    metrics: Dict[str, float],
) -> bool:
    timestamp = metrics[metric_name]
    now = time.time()
    idle_duration = now - timestamp
    if idle_duration > config.idle:
        logger.info(
            '%s watch [%s]: %s=%.3f, idle=%.1fs > threshold=%ss,'
            ' executing action=%s',
            watch_name,
            buffer,
            metric_name,
            timestamp,
            idle_duration,
            config.idle,
            config.action.value,
        )
        return True
    logger.info(
        '%s watch [%s]: %s=%.3f, idle=%.1fs <= threshold=%ss, no action',
        watch_name,
        buffer,
        metric_name,
        timestamp,
        idle_duration,
        config.idle,
    )
    return False


async def _watch_loop(
    docker_client: DockerClient,
    buffer: str,
    watch_name: str,
    config: Union[QueueConfig, FlowConfig],
    evaluate: Callable[[Dict[str, float]], bool],
):
    await asyncio.sleep(config.polling_interval)

    while True:
        try:
            content = await get_metrics(buffer)
            metrics = await parse_metrics(content, config.label_filters)

            if evaluate(metrics):
                await process_action(
                    docker_client, config.action, config.container_labels
                )
                await asyncio.sleep(config.cooldown)
            else:
                await asyncio.sleep(config.polling_interval)

        except KeyError as e:
            logger.warning(
                '%s watch [%s]: metric %s not found in response, skipping cycle',
                watch_name,
                buffer,
                e,
            )
            await asyncio.sleep(config.polling_interval)
        except Exception as e:
            logger.warning(
                '%s watch [%s]: %s: %s, skipping cycle',
                watch_name,
                buffer,
                type(e).__name__,
                e,
            )
            await asyncio.sleep(config.polling_interval)


async def watch_queue(docker_client: DockerClient, buffer: str, config: QueueConfig):
    await _watch_loop(
        docker_client,
        buffer,
        'Queue',
        config,
        lambda metrics: _queue_evaluate(buffer, config, metrics),
    )


async def watch_egress(docker_client: DockerClient, buffer: str, config: FlowConfig):
    await _watch_loop(
        docker_client,
        buffer,
        'Egress',
        config,
        lambda metrics: _flow_evaluate(
            'Egress', LAST_SENT_MESSAGE_METRIC, buffer, config, metrics
        ),
    )


async def watch_ingress(docker_client: DockerClient, buffer: str, config: FlowConfig):
    await _watch_loop(
        docker_client,
        buffer,
        'Ingress',
        config,
        lambda metrics: _flow_evaluate(
            'Ingress', LAST_RECEIVED_MESSAGE_METRIC, buffer, config, metrics
        ),
    )


async def watch_buffer(docker_client: DockerClient, config: WatchConfig):
    logger.info('Watching buffer [%s] metrics', config.buffer)
    watches = []

    if config.queue:
        logger.info('Watching queue: %s', config.queue)
        watches.append(watch_queue(docker_client, config.buffer, config.queue))
    if config.egress:
        logger.info('Watching egress flow: %s', config.egress)
        watches.append(watch_egress(docker_client, config.buffer, config.egress))
    if config.ingress:
        logger.info('Watching ingress flow: %s', config.ingress)
        watches.append(watch_ingress(docker_client, config.buffer, config.ingress))

    await asyncio.gather(*watches)


async def watch(config: Config):
    docker_client = DockerClient()

    await asyncio.gather(
        *[watch_buffer(docker_client, x) for x in config.watch_configs]
    )

    await docker_client.close()


def main():
    # To gracefully shut down on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    config_file_path = os.environ.get('CONFIG_FILE_PATH')
    if not config_file_path:
        logger.error(
            'Configuration file path is not provided. '
            'Provide the CONFIG_FILE_PATH environment variable'
        )
        exit(1)

    parser = ConfigParser(config_file_path)

    try:
        config = parser.parse()
        validate(config)
    except Exception as e:
        logger.error('Invalid configuration. %s: %s', type(e).__name__, e)
        exit(1)

    asyncio.run(watch(config))
