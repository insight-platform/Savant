import asyncio
import time
from unittest import mock
from unittest.mock import call

import pytest
from watchdog.main import watch_buffer, watch_pyfunc


# --- watch_pyfunc tests ---


@pytest.mark.asyncio
@mock.patch('watchdog.main.process_action')
@mock.patch('watchdog.main.get_metrics', return_value='content')
@mock.patch(
    'watchdog.main.parse_metrics',
    return_value={'last_sent_message': time.time() - 120},
)
@mock.patch('watchdog.main._load_pyfunc')
@mock.patch('watchdog.main.DockerClient')
async def test_watch_pyfunc_trigger_fires(
    docker_client_mock,
    load_pyfunc_mock,
    parse_metrics_mock,
    get_metrics_mock,
    process_action_mock,
    pyfunc_config,
):
    """When pyfunc returns True, action is executed and cooldown is used."""
    docker_client = docker_client_mock()
    trigger = mock.Mock(return_value=True)
    load_pyfunc_mock.return_value = trigger

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_pyfunc(docker_client, 'buffer1:8000', pyfunc_config)
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(pyfunc_config.polling_interval),
                    call(pyfunc_config.cooldown),
                ]
            )

    trigger.assert_called_once()
    process_action_mock.assert_awaited_once_with(
        docker_client, pyfunc_config.action, pyfunc_config.container_labels
    )


@pytest.mark.asyncio
@mock.patch('watchdog.main.process_action')
@mock.patch('watchdog.main.get_metrics', return_value='content')
@mock.patch(
    'watchdog.main.parse_metrics',
    return_value={'last_sent_message': time.time()},
)
@mock.patch('watchdog.main._load_pyfunc')
@mock.patch('watchdog.main.DockerClient')
async def test_watch_pyfunc_trigger_does_not_fire(
    docker_client_mock,
    load_pyfunc_mock,
    parse_metrics_mock,
    get_metrics_mock,
    process_action_mock,
    pyfunc_config,
):
    """When pyfunc returns False, no action and polling_interval is used."""
    docker_client = docker_client_mock()
    trigger = mock.Mock(return_value=False)
    load_pyfunc_mock.return_value = trigger

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_pyfunc(docker_client, 'buffer1:8000', pyfunc_config)
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(pyfunc_config.polling_interval),
                    call(pyfunc_config.polling_interval),
                ]
            )

    trigger.assert_called_once()
    process_action_mock.assert_not_awaited()


@pytest.mark.asyncio
@mock.patch('watchdog.main.process_action')
@mock.patch('watchdog.main.get_metrics', return_value='content')
@mock.patch('watchdog.main.parse_metrics', return_value={})
@mock.patch('watchdog.main._load_pyfunc')
@mock.patch('watchdog.main.DockerClient')
async def test_watch_pyfunc_missing_metric_skips_cycle(
    docker_client_mock,
    load_pyfunc_mock,
    parse_metrics_mock,
    get_metrics_mock,
    process_action_mock,
    pyfunc_config,
):
    """KeyError from pyfunc (missing metric) should skip cycle."""
    docker_client = docker_client_mock()
    trigger = mock.Mock(side_effect=KeyError('last_sent_message'))
    load_pyfunc_mock.return_value = trigger

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_pyfunc(docker_client, 'buffer1:8000', pyfunc_config)
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(pyfunc_config.polling_interval),
                    call(pyfunc_config.polling_interval),
                ]
            )

    process_action_mock.assert_not_awaited()


@pytest.mark.asyncio
@mock.patch('watchdog.main.process_action')
@mock.patch('watchdog.main.get_metrics', side_effect=Exception('connection failed'))
@mock.patch('watchdog.main._load_pyfunc')
@mock.patch('watchdog.main.DockerClient')
async def test_watch_pyfunc_http_error_skips_cycle(
    docker_client_mock,
    load_pyfunc_mock,
    get_metrics_mock,
    process_action_mock,
    pyfunc_config,
):
    """HTTP failure should skip cycle, not crash."""
    docker_client = docker_client_mock()
    trigger = mock.Mock()
    load_pyfunc_mock.return_value = trigger

    with mock.patch(
        'asyncio.sleep', side_effect=[None, asyncio.CancelledError]
    ) as sleep_mock:
        try:
            await watch_pyfunc(docker_client, 'buffer1:8000', pyfunc_config)
        except asyncio.CancelledError:
            sleep_mock.assert_has_awaits(
                [
                    call(pyfunc_config.polling_interval),
                    call(pyfunc_config.polling_interval),
                ]
            )

    trigger.assert_not_called()
    process_action_mock.assert_not_awaited()


# --- watch_buffer integration with pyfunc ---


@pytest.mark.asyncio
@mock.patch('watchdog.main.watch_pyfunc')
@mock.patch('watchdog.main.watch_ingress')
@mock.patch('watchdog.main.watch_egress')
@mock.patch('watchdog.main.watch_queue')
@mock.patch('watchdog.main.DockerClient')
async def test_watch_buffer_pyfunc_only(
    docker_client_mock,
    watch_queue_mock,
    watch_egress_mock,
    watch_ingress_mock,
    watch_pyfunc_mock,
    config_with_pyfunc_only,
):
    docker_client = docker_client_mock()
    watch_config = config_with_pyfunc_only.watch_configs[0]

    await watch_buffer(docker_client, watch_config)

    watch_pyfunc_mock.assert_awaited_once_with(
        docker_client, watch_config.buffer, watch_config.pyfunc
    )
    watch_queue_mock.assert_not_awaited()
    watch_egress_mock.assert_not_awaited()
    watch_ingress_mock.assert_not_awaited()
