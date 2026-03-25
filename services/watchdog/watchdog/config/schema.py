"""The configuration classes for the pipeline watchdog."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


def validate_container_labels(labels: List[List[str]]):
    if not labels:
        raise ValueError('Container labels cannot be empty.')


class Action(Enum):
    STOP = 'stop'
    RESTART = 'restart'


@dataclass
class QueueConfig:
    """Configuration to watch a buffer queue."""

    action: Action
    """Action to take when buffer queue is full."""

    length: int
    """Maximum buffer queue length."""

    cooldown: int
    """Interval in seconds to wait after applying the action."""

    polling_interval: int
    """Interval in seconds between buffer queue length checks."""

    container_labels: List[List[str]]
    """List of labels to filter the containers to which the action is applied."""

    label_filters: Optional[Dict[str, Dict[str, str]]] = field(default=None)
    """Optional mapping of metric name to label key=value pairs.

    When set, only metric samples whose labels match all specified pairs
    are considered.  When ``None``, ``max()`` across all samples of the
    same metric name is used.
    """

    def __post_init__(self):
        validate_container_labels(self.container_labels)


@dataclass
class FlowConfig:
    """Configuration to watch a buffer incoming or outgoing traffic."""

    action: Action
    """Action to take when buffer traffic is idle."""

    idle: int
    """Maximum time in seconds buffer traffic can be idle."""

    cooldown: int
    """Interval in seconds to wait after applying the action."""

    polling_interval: Optional[int]
    """Interval in seconds between buffer traffic checks."""

    container_labels: List[List[str]]
    """List of labels to filter the containers to which the action is applied."""

    label_filters: Optional[Dict[str, Dict[str, str]]] = field(default=None)
    """Optional mapping of metric name to label key=value pairs.

    When set, only metric samples whose labels match all specified pairs
    are considered.  When ``None``, ``max()`` across all samples of the
    same metric name is used.
    """

    def __post_init__(self):
        validate_container_labels(self.container_labels)


@dataclass
class PyFuncConfig:
    """Configuration for a custom pyfunc watch trigger."""

    action: Action
    """Action to take when the pyfunc returns True."""

    cooldown: int
    """Interval in seconds to wait after applying the action."""

    polling_interval: int
    """Interval in seconds between trigger checks."""

    container_labels: List[List[str]]
    """List of labels to filter the containers to which the action is applied."""

    module: str
    """Python module path to import (e.g. 'my_checks.discrepancy')."""

    class_name: str
    """Class name within the module. Must be callable (implement __call__)."""

    kwargs: Optional[Dict] = field(default=None)
    """Optional keyword arguments passed to the class constructor."""

    def __post_init__(self):
        validate_container_labels(self.container_labels)


@dataclass
class WatchConfig:
    """Configuration for a single buffer."""

    buffer: str
    """Buffer url to retrieve metrics."""

    queue: Optional[QueueConfig]
    """Queue watch configuration."""

    egress: Optional[FlowConfig]
    """Egress traffic watch configuration."""

    ingress: Optional[FlowConfig]
    """Ingress traffic watch configuration."""

    pyfunc: Optional[PyFuncConfig] = None
    """Custom pyfunc watch configuration."""


@dataclass
class Config:
    """Pipeline watchdog configuration."""

    watch_configs: List[WatchConfig]
    """List of buffer watch configurations."""
