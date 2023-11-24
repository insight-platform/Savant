from abc import ABC, abstractmethod

from savant.metrics.metric import Metric


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
