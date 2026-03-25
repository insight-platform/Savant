import time
from unittest import mock

import pytest
from watchdog.triggers.discrepancy import DiscrepancyCheck


@pytest.fixture
def check():
    return DiscrepancyCheck(
        buffer_url='buffer:8000',
        egress_idle=60,
        ingress_idle=30,
    )


@pytest.mark.asyncio
@mock.patch('watchdog.triggers.discrepancy.get_metrics', return_value='content')
@mock.patch(
    'watchdog.triggers.discrepancy.parse_metrics',
    return_value={
        'last_sent_message': time.time() - 120,  # egress idle 120s > 60s threshold
        'last_received_message': time.time() - 5,  # ingress active 5s < 30s threshold
    },
)
async def test_discrepancy_module_stuck(parse_mock, get_mock, check):
    """Egress idle + ingress active = module stuck -> trigger."""
    assert await check() is True
    get_mock.assert_awaited_once_with('buffer:8000')


@pytest.mark.asyncio
@mock.patch('watchdog.triggers.discrepancy.get_metrics', return_value='content')
@mock.patch(
    'watchdog.triggers.discrepancy.parse_metrics',
    return_value={
        'last_sent_message': time.time() - 120,  # egress idle
        'last_received_message': time.time() - 120,  # ingress also idle
    },
)
async def test_discrepancy_both_idle(parse_mock, get_mock, check):
    """Both egress and ingress idle = upstream problem, not stuck -> no trigger."""
    assert await check() is False


@pytest.mark.asyncio
@mock.patch('watchdog.triggers.discrepancy.get_metrics', return_value='content')
@mock.patch(
    'watchdog.triggers.discrepancy.parse_metrics',
    return_value={
        'last_sent_message': time.time() - 5,  # egress active
        'last_received_message': time.time() - 5,  # ingress active
    },
)
async def test_discrepancy_both_active(parse_mock, get_mock, check):
    """Both active = healthy -> no trigger."""
    assert await check() is False


@pytest.mark.asyncio
@mock.patch('watchdog.triggers.discrepancy.get_metrics', return_value='content')
@mock.patch(
    'watchdog.triggers.discrepancy.parse_metrics',
    return_value={
        'last_sent_message': time.time() - 5,  # egress active
        'last_received_message': time.time() - 120,  # ingress idle
    },
)
async def test_discrepancy_egress_active_ingress_idle(parse_mock, get_mock, check):
    """Egress active, ingress idle = draining, not stuck -> no trigger."""
    assert await check() is False


@pytest.mark.asyncio
@mock.patch('watchdog.triggers.discrepancy.get_metrics', return_value='content')
@mock.patch(
    'watchdog.triggers.discrepancy.parse_metrics',
    return_value={
        'last_sent_message': time.time() - 120,
    },
)
async def test_discrepancy_missing_ingress_metric(parse_mock, get_mock, check):
    """Missing ingress metric -> no trigger (safe fallback)."""
    assert await check() is False


@pytest.mark.asyncio
@mock.patch('watchdog.triggers.discrepancy.get_metrics', return_value='content')
@mock.patch(
    'watchdog.triggers.discrepancy.parse_metrics',
    return_value={},
)
async def test_discrepancy_missing_both_metrics(parse_mock, get_mock, check):
    """Missing both metrics -> no trigger."""
    assert await check() is False
