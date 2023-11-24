import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from savant.utils.logging import get_logger

logger = get_logger(__name__)


class Metric:
    """Base class for metrics.

    :param name: Metric name.
    :param description: Metric description.
    :param labelnames: Metric label names.
    """

    def __init__(
        self,
        name: str,
        description: str = '',
        labelnames: Tuple[str, ...] = (),
    ):
        self._name = name
        self._description = description or name
        self._labelnames = labelnames
        self._values: Dict[Tuple[str, ...], Tuple[float, float]] = {}

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def labelnames(self):
        return self._labelnames

    @property
    def values(self):
        return self._values


class Counter(Metric):
    def inc(
        self,
        amount=1,
        labels: Tuple[str, ...] = (),
        timestamp: Optional[float] = None,
    ):
        """Increment counter by amount.

        :param amount: Increment amount.
        :param labels: Labels values.
        :param timestamp: Metric timestamp.
        """

        assert len(labels) == len(self._labelnames), 'Labels must match label names'
        assert amount > 0, 'Counter increment amount must be positive'
        last_value = self._values.get(labels, (0, 0))[0]
        if timestamp is None:
            timestamp = time.time()
        self._values[labels] = last_value + amount, timestamp

    def set(
        self,
        value,
        labels: Tuple[str, ...] = (),
        timestamp: Optional[float] = None,
    ):
        """Set counter to specific value.

        :param value: Counter value. Must be non-decreasing.
        :param labels: Labels values.
        :param timestamp: Metric timestamp.
        """

        assert len(labels) == len(self._labelnames), 'Labels must match label names'
        last_value = self._values.get(labels, (0, 0))[0]
        assert value >= last_value, 'Counter value must be non-decreasing'
        if timestamp is None:
            timestamp = time.time()
        self._values[labels] = value, timestamp


class Gauge(Metric):
    def set(
        self,
        value,
        labels: Tuple[str, ...] = (),
        timestamp: Optional[float] = None,
    ):
        """Set gauge to specific value.

        :param value: Gauge value.
        :param labels: Labels values.
        :param timestamp: Metric timestamp.
        """

        assert len(labels) == len(self._labelnames), 'Labels must match label names'
        if timestamp is None:
            timestamp = time.time()
        self._values[labels] = value, timestamp


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
