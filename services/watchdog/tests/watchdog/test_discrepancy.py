import time

import pytest
from watchdog.triggers.discrepancy import DiscrepancyCheck


@pytest.fixture
def check():
    return DiscrepancyCheck(egress_idle=60, ingress_idle=30)


def test_discrepancy_module_stuck(check):
    """Egress idle + ingress active = module stuck -> trigger."""
    metrics = {
        'last_sent_message': time.time() - 120,  # egress idle 120s > 60s
        'last_received_message': time.time() - 5,  # ingress active 5s < 30s
    }
    assert check(metrics) is True


def test_discrepancy_both_idle(check):
    """Both egress and ingress idle = upstream problem, not stuck -> no trigger."""
    metrics = {
        'last_sent_message': time.time() - 120,
        'last_received_message': time.time() - 120,
    }
    assert check(metrics) is False


def test_discrepancy_both_active(check):
    """Both active = healthy -> no trigger."""
    metrics = {
        'last_sent_message': time.time() - 5,
        'last_received_message': time.time() - 5,
    }
    assert check(metrics) is False


def test_discrepancy_egress_active_ingress_idle(check):
    """Egress active, ingress idle = draining, not stuck -> no trigger."""
    metrics = {
        'last_sent_message': time.time() - 5,
        'last_received_message': time.time() - 120,
    }
    assert check(metrics) is False


def test_discrepancy_missing_egress_metric(check):
    """Missing egress metric -> KeyError (handled by watch loop)."""
    metrics = {'last_received_message': time.time() - 5}
    with pytest.raises(KeyError):
        check(metrics)


def test_discrepancy_missing_both_metrics(check):
    """Missing both metrics -> KeyError."""
    with pytest.raises(KeyError):
        check({})
