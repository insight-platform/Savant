import pytest
from watchdog.config.validator import validate


@pytest.mark.parametrize(
    'config1',
    [
        'config',
        'config_with_queue_only',
        'config_with_ingress_only',
        'config_with_egress_only',
    ],
)
def test_validate(request, config1):
    validate(request.getfixturevalue(config1))


def test_validate_empty_watch(config_with_invalid_watch_config):
    with pytest.raises(
        ValueError,
        match='Watch config must include at least one of the following: queue, ingress, or egress.',
    ):
        validate(config_with_invalid_watch_config)
