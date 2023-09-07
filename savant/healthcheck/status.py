from enum import Enum


class PipelineStatus(Enum):
    """Pipeline status."""

    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'
    STOPPED = 'stopped'
