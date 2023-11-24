from abc import ABC, abstractmethod
from typing import Optional

from savant.metrics.metric import Metric
from savant.utils.logging import get_logger

logger = get_logger(__name__)


class BaseMetricsExporter(ABC):
    """Base class for metrics exporters."""

    @abstractmethod
    def start(self):
        """Start metrics exporter."""

    @abstractmethod
    def stop(self):
        """Stop metrics exporter."""

    @abstractmethod
    def register_metric(self, metric: Metric):
        """Register metric."""
        pass


class MetricsRegistry:
    def __init__(self, exporter: Optional[BaseMetricsExporter]):
        self._exporter = exporter
        self._metrics = {}

    def __setitem__(self, key, value: Metric):
        if not isinstance(value, Metric):
            raise ValueError('Value must be a Metric instance')
        if key in self._metrics:
            raise KeyError(f'Key {key!r} already exists')
        if self._exporter is not None:
            self._exporter.register_metric(value)
        else:
            logger.warning(
                'Metric exporter not configured. Ignoring metric %s.',
                value.name,
            )
        self._metrics[key] = value

    def __getitem__(self, key):
        return self._metrics[key]

    def __contains__(self, key):
        return key in self._metrics
