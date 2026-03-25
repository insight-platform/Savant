import os

import pytest
import yaml
from watchdog.config.parser import ConfigParser
from watchdog.config.schema import Action


@pytest.fixture(scope='session')
def pyfunc_config_file_path():
    return os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'test_pyfunc_config.yml'
    )


def test_parse_pyfunc(pyfunc_config_file_path):
    config = ConfigParser(pyfunc_config_file_path).parse()

    assert len(config.watch_configs) == 1
    wc = config.watch_configs[0]
    assert wc.buffer == 'buffer1:8000'
    assert wc.queue is None
    assert wc.egress is None
    assert wc.ingress is None
    assert wc.pyfunc is not None

    pf = wc.pyfunc
    assert pf.action == Action.RESTART
    assert pf.cooldown == 120
    assert pf.polling_interval == 10
    assert pf.container_labels == [['com.savant.module=detector']]
    assert pf.module == 'watchdog.triggers.discrepancy'
    assert pf.class_name == 'DiscrepancyCheck'
    assert pf.kwargs == {
        'egress_idle': 60,
        'ingress_idle': 30,
    }


def test_parse_pyfunc_bad_module(tmpdir):
    config_data = {
        'watch': [
            {
                'buffer': 'buffer:8000',
                'pyfunc': {
                    'action': 'restart',
                    'cooldown': '10s',
                    'polling_interval': '5s',
                    'container': [{'labels': ['some-label']}],
                    'module': 'nonexistent.module',
                    'class_name': 'Foo',
                },
            }
        ]
    }
    config_file = tmpdir.join('bad_module.yml')
    config_file.write(yaml.dump(config_data))

    with pytest.raises(ValueError, match='Failed to import pyfunc module'):
        ConfigParser(str(config_file)).parse()


def test_parse_pyfunc_bad_class(tmpdir):
    config_data = {
        'watch': [
            {
                'buffer': 'buffer:8000',
                'pyfunc': {
                    'action': 'restart',
                    'cooldown': '10s',
                    'polling_interval': '5s',
                    'container': [{'labels': ['some-label']}],
                    'module': 'watchdog.triggers.discrepancy',
                    'class_name': 'NonexistentClass',
                },
            }
        ]
    }
    config_file = tmpdir.join('bad_class.yml')
    config_file.write(yaml.dump(config_data))

    with pytest.raises(ValueError, match='Class "NonexistentClass" not found'):
        ConfigParser(str(config_file)).parse()


def test_parse_pyfunc_empty_labels(tmpdir):
    config_data = {
        'watch': [
            {
                'buffer': 'buffer:8000',
                'pyfunc': {
                    'action': 'restart',
                    'cooldown': '10s',
                    'polling_interval': '5s',
                    'container': [],
                    'module': 'watchdog.triggers.discrepancy',
                    'class_name': 'DiscrepancyCheck',
                },
            }
        ]
    }
    config_file = tmpdir.join('empty_labels.yml')
    config_file.write(yaml.dump(config_data))

    with pytest.raises(ValueError, match='Container labels cannot be empty'):
        ConfigParser(str(config_file)).parse()
