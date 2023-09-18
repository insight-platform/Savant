from enum import Enum
from pathlib import Path

from savant.utils.logging import get_logger

logger = get_logger(__name__)


class ModuleStatus(Enum):
    """Module status."""

    INITIALIZING = 'initializing'
    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'
    STOPPED = 'stopped'


def set_module_status(status_filepath: Path, status: ModuleStatus):
    logger.info('Setting module status to %s.', status)
    status_filepath.write_text(f'{status.value}\n')
