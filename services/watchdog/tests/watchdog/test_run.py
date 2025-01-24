import asyncio
import time
from unittest import mock
from unittest.mock import AsyncMock, call

import pytest
from aiodocker.containers import DockerContainer
from watchdog.config.schema import Action
from watchdog.run import (
    main,
    process_action,
    watch_buffer,
    watch_egress,
    watch_ingress,
    watch_queue,
)


@pytest.mark.asyncio
@mock.patch('watchdog.run.DockerClient', autospec=True)
async def test_process_action_stop(docker_client_mock):
    docker_client = docker_client_mock()
    docker_container1 = AsyncMock(DockerContainer)
    docker_container2 = AsyncMock(DockerContainer)
    docker_client.get_containers = mock.AsyncMock(
        return_value=[docker_container1, docker_container2]
    )
    container_labels = [['label1']]

    await process_action(docker_client, Action.STOP, container_labels)

    docker_client.get_containers.assert_awaited_once_with(container_labels)
    docker_client.stop_container.assert_has_awaits(
        [call(docker_container1), call(docker_container2)]
    )


@pytest.mark.asyncio
@mock.patch('watchdog.run.DockerClient', autospec=True)
async def test_process_action_restart(docker_client_mock):
    docker_client = docker_client_mock()
    docker_container1 = AsyncMock(DockerContainer)
    docker_container2 = AsyncMock(DockerContainer)
    docker_client.get_containers = mock.AsyncMock(
        return_value=[docker_container1, docker_container2]
    )
    container_labels = [['label1']]

    await process_action(docker_client, Action.RESTART, container_labels)

    docker_client.get_containers.assert_awaited_once_with(container_labels)
    docker_client.restart_container.assert_has_awaits(
        [call(docker_container1), call(docker_container2)]
    )


@pytest.mark.asyncio
@mock.patch('watchdog.run.DockerClient', autospec=True)
@pytest.mark.parametrize(
    'action',
    [
        Action.STOP,
        Action.RESTART,
    ],
)
async def test_process_action_empty_containers(docker_client_mock, action):
    docker_client = docker_client_mock()
    docker_client.get_containers = mock.AsyncMock(return_value=[])
    container_labels = [['label1']]

    await process_action(docker_client, action, container_labels)

    docker_client.get_containers.assert_awaited_once_with(container_labels)


@pytest.mark.asyncio
@mock.patch('watchdog.run.DockerClient', autospec=True)
async def test_process_action_invalid(docker_client_mock):
    docker_client = docker_client_mock()

    with pytest.raises(RuntimeError, match='Unknown action: invalid_action'):
        await process_action(docker_client, 'invalid_action', [])  # type: ignore


@pytest.mark.asyncio
@mock.patch('watchdog.run.watch_ingress')
@mock.patch('watchdog.run.watch_egress')
@mock.patch('watchdog.run.watch_queue')
@mock.patch('watchdog.run.DockerClient')
async def test_watch_buffer(
    docker_client_mock,
    watch_queue_mock,
    watch_egress_mock,
    watch_ingress_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    await watch_buffer(docker_client, watch_config)
    watch_queue_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.queue
    )
    watch_egress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.egress
    )
    watch_ingress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.ingress
    )


@pytest.mark.asyncio
@mock.patch('watchdog.run.watch_ingress')
@mock.patch('watchdog.run.watch_egress')
@mock.patch('watchdog.run.watch_queue')
@mock.patch('watchdog.run.DockerClient')
async def test_watch_buffer_queue_only(
    docker_client_mock,
    watch_queue_mock,
    watch_egress_mock,
    watch_ingress_mock,
    config_with_queue_only,
):
    docker_client = docker_client_mock()

    watch_config = config_with_queue_only.watch_configs[0]

    await watch_buffer(docker_client, watch_config)
    watch_queue_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.queue
    )
    watch_egress_mock.assert_not_awaited()
    watch_ingress_mock.assert_not_awaited()


@pytest.mark.asyncio
@mock.patch('watchdog.run.watch_ingress')
@mock.patch('watchdog.run.watch_egress')
@mock.patch('watchdog.run.watch_queue')
@mock.patch('watchdog.run.DockerClient')
async def test_watch_buffer_egress_only(
    docker_client_mock,
    watch_queue_mock,
    watch_egress_mock,
    watch_ingress_mock,
    config_with_egress_only,
):
    docker_client = docker_client_mock()

    watch_config = config_with_egress_only.watch_configs[0]

    await watch_buffer(docker_client, watch_config)
    watch_egress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.egress
    )
    watch_queue_mock.assert_not_awaited()
    watch_ingress_mock.assert_not_awaited()


@pytest.mark.asyncio
@mock.patch('watchdog.run.watch_ingress')
@mock.patch('watchdog.run.watch_egress')
@mock.patch('watchdog.run.watch_queue')
@mock.patch('watchdog.run.DockerClient')
async def test_watch_buffer_ingress_only(
    docker_client_mock,
    watch_queue_mock,
    watch_egress_mock,
    watch_ingress_mock,
    config_with_ingress_only,
):
    docker_client = docker_client_mock()

    watch_config = config_with_ingress_only.watch_configs[0]

    await watch_buffer(docker_client, watch_config)
    watch_ingress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.ingress
    )
    watch_queue_mock.assert_not_awaited()
    watch_egress_mock.assert_not_awaited()


@pytest.mark.asyncio
@mock.patch('watchdog.run.watch_ingress')
@mock.patch('watchdog.run.watch_egress')
@mock.patch('watchdog.run.watch_queue', side_effect=RuntimeError('error'))
@mock.patch('watchdog.run.DockerClient')
async def test_watch_buffer_watch_queue_failed(
    docker_client_mock,
    watch_queue_mock,
    watch_egress_mock,
    watch_ingress_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    with pytest.raises(RuntimeError, match='error'):
        await watch_buffer(docker_client, watch_config)

    watch_queue_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.queue
    )
    watch_egress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.egress
    )
    watch_ingress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.ingress
    )


@pytest.mark.asyncio
@mock.patch('watchdog.run.watch_ingress')
@mock.patch('watchdog.run.watch_egress', side_effect=RuntimeError('error'))
@mock.patch('watchdog.run.watch_queue')
@mock.patch('watchdog.run.DockerClient')
async def test_watch_buffer_watch_egress_failed(
    docker_client_mock,
    watch_queue_mock,
    watch_egress_mock,
    watch_ingress_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    with pytest.raises(RuntimeError, match='error'):
        await watch_buffer(docker_client, watch_config)

    watch_queue_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.queue
    )
    watch_egress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.egress
    )
    watch_ingress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.ingress
    )


@pytest.mark.asyncio
@mock.patch('watchdog.run.watch_ingress', side_effect=RuntimeError('error'))
@mock.patch('watchdog.run.watch_egress')
@mock.patch('watchdog.run.watch_queue')
@mock.patch('watchdog.run.DockerClient')
async def test_watch_buffer_watch_ingress_failed(
    docker_client_mock,
    watch_queue_mock,
    watch_egress_mock,
    watch_ingress_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    with pytest.raises(RuntimeError, match='error'):
        await watch_buffer(docker_client, watch_config)

    watch_queue_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.queue
    )
    watch_egress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.egress
    )
    watch_ingress_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.ingress
    )


@pytest.mark.asyncio
@mock.patch('watchdog.run.process_action')
@mock.patch('watchdog.run.get_metrics', return_value='content')
@mock.patch('watchdog.run.parse_metrics', return_value={'buffer_size': 999})
@mock.patch('watchdog.run.DockerClient')
async def test_watch_queue(
    docker_client_mock,
    parse_metrics_mock,
    get_metrics_mock,
    process_action_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_queue(docker_client, watch_config.buffer, watch_config.queue)
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(watch_config.queue.polling_interval),
                    call(watch_config.queue.cooldown),
                ]
            )
            pass

    get_metrics_mock.assert_awaited_once_with(watch_config.buffer)
    parse_metrics_mock.assert_awaited_once_with('content')
    process_action_mock.assert_awaited_once_with(
        docker_client, watch_config.queue.action, watch_config.queue.container_labels
    )


@pytest.mark.asyncio
@mock.patch('watchdog.run.process_action')
@mock.patch('watchdog.run.get_metrics', return_value='content')
@mock.patch('watchdog.run.parse_metrics', return_value={'buffer_size': 0})
@mock.patch('watchdog.run.DockerClient')
async def test_watch_queue_empty(
    docker_client_mock,
    parse_metrics_mock,
    get_metrics_mock,
    process_action_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_queue(docker_client, watch_config.buffer, watch_config.queue)
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(watch_config.queue.polling_interval),
                    call(watch_config.queue.polling_interval),
                ]
            )
            pass

    get_metrics_mock.assert_awaited_once_with(watch_config.buffer)
    parse_metrics_mock.assert_awaited_once_with('content')
    process_action_mock.assert_not_awaited()


@pytest.mark.asyncio
@mock.patch('watchdog.run.process_action')
@mock.patch('watchdog.run.get_metrics', return_value='content')
@mock.patch(
    'watchdog.run.parse_metrics',
    return_value={'last_sent_message': time.time() - 999},
)
@mock.patch('watchdog.run.DockerClient')
async def test_watch_egress(
    docker_client_mock,
    parse_metrics_mock,
    get_metrics_mock,
    process_action_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_egress(docker_client, watch_config.buffer, watch_config.egress)
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(watch_config.egress.polling_interval),
                    call(watch_config.egress.cooldown),
                ]
            )
            pass

    get_metrics_mock.assert_awaited_once_with(watch_config.buffer)
    parse_metrics_mock.assert_awaited_once_with('content')
    process_action_mock.assert_awaited_once_with(
        docker_client, watch_config.egress.action, watch_config.egress.container_labels
    )


@pytest.mark.asyncio
@mock.patch('watchdog.run.process_action')
@mock.patch('watchdog.run.get_metrics', return_value='content')
@mock.patch(
    'watchdog.run.parse_metrics',
    return_value={'last_sent_message': time.time()},
)
@mock.patch('watchdog.run.DockerClient')
async def test_watch_egress_message_just_sent(
    docker_client_mock,
    parse_metrics_mock,
    get_metrics_mock,
    process_action_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_egress(docker_client, watch_config.buffer, watch_config.egress)
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(watch_config.egress.polling_interval),
                    call(watch_config.egress.polling_interval),
                ]
            )
            pass

    get_metrics_mock.assert_awaited_once_with(watch_config.buffer)
    parse_metrics_mock.assert_awaited_once_with('content')
    process_action_mock.assert_not_awaited()


@pytest.mark.asyncio
@mock.patch('watchdog.run.process_action')
@mock.patch('watchdog.run.get_metrics', return_value='content')
@mock.patch(
    'watchdog.run.parse_metrics',
    return_value={'last_received_message': time.time() - 999},
)
@mock.patch('watchdog.run.DockerClient')
async def test_watch_ingress(
    docker_client_mock,
    parse_metrics_mock,
    get_metrics_mock,
    process_action_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_ingress(
                docker_client, watch_config.buffer, watch_config.ingress
            )
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(watch_config.ingress.polling_interval),
                    call(watch_config.ingress.cooldown),
                ]
            )
            pass

    get_metrics_mock.assert_awaited_once_with(watch_config.buffer)
    parse_metrics_mock.assert_awaited_once_with('content')
    process_action_mock.assert_awaited_once_with(
        docker_client,
        watch_config.ingress.action,
        watch_config.ingress.container_labels,
    )


@pytest.mark.asyncio
@mock.patch('watchdog.run.process_action')
@mock.patch('watchdog.run.get_metrics', return_value='content')
@mock.patch(
    'watchdog.run.parse_metrics',
    return_value={'last_received_message': time.time()},
)
@mock.patch('watchdog.run.DockerClient')
async def test_watch_ingress_message_just_received(
    docker_client_mock,
    parse_metrics_mock,
    get_metrics_mock,
    process_action_mock,
    watch_config,
):
    docker_client = docker_client_mock()

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_ingress(
                docker_client, watch_config.buffer, watch_config.ingress
            )
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(watch_config.ingress.polling_interval),
                    call(watch_config.ingress.polling_interval),
                ]
            )
            pass

    get_metrics_mock.assert_awaited_once_with(watch_config.buffer)
    parse_metrics_mock.assert_awaited_once_with('content')
    process_action_mock.assert_not_awaited()


@pytest.mark.parametrize('config', [None, ''])
@mock.patch('os.environ.get')
def test_main_no_config_file_path(environ_mock, config):
    environ_mock.return_value = config

    with pytest.raises(SystemExit, match='1'):
        main()


@mock.patch('watchdog.run.validate', side_effect=RuntimeError('error'))
@mock.patch('watchdog.run.ConfigParser')
@mock.patch('os.environ.get', return_value='config.yml')
def test_main_invalid_config(environ_mock, config_parser_mock, validate_mock):
    config_parser = config_parser_mock.return_value

    with pytest.raises(SystemExit, match='1'):
        main()

    config_parser_mock.assert_called_once_with('config.yml')
    config_parser.parse.assert_called_once()
    parsed_config = config_parser.parse.return_value
    validate_mock.assert_called_once_with(parsed_config)


@mock.patch('watchdog.run.watch_buffer')
@mock.patch('watchdog.run.DockerClient', autospec=True)
@mock.patch('watchdog.run.validate')
@mock.patch('watchdog.run.ConfigParser')
@mock.patch('os.environ.get', return_value='config.yml')
def test_main_keyboard_interrupt(
    environ_mock,
    config_parser_mock,
    validate_mock,
    docker_client_mock,
    watch_buffer_mock,
    config,
):
    config_parser = config_parser_mock.return_value
    config_parser.parse.return_value = config

    def raise_exception():
        raise KeyboardInterrupt()

    loop = asyncio.new_event_loop()
    loop.call_later(1, raise_exception)
    asyncio.set_event_loop(loop)

    main()

    assert loop.is_closed()
    docker_client_mock.return_value.close.assert_awaited_once()

    config_parser_mock.assert_called_once_with('config.yml')
    config_parser.parse.assert_called_once()
    validate_mock.assert_called_once_with(config)
    watch_buffer_mock.assert_awaited_once_with(
        docker_client_mock.return_value, config.watch_configs[0]
    )
