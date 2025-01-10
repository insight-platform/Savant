from abc import ABC, abstractmethod


class BaseMetricsExporter(ABC):
    """Base class for metrics exporters."""

    @abstractmethod
    def start(self):
        """Start metrics exporter."""

    @abstractmethod
    def stop(self):
        """Stop metrics exporter."""
