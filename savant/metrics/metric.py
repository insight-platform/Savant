import time
from typing import Dict, Optional, Tuple


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
    def name(self) -> str:
        """Metric name."""
        return self._name

    @property
    def description(self) -> str:
        """Metric description."""
        return self._description

    @property
    def labelnames(self) -> Tuple[str, ...]:
        """Metric label names."""
        return self._labelnames

    @property
    def values(self) -> Dict[Tuple[str, ...], Tuple[float, float]]:
        """Metric values.

        :return: Dictionary: labels -> (value, timestamp).
        """
        return self._values


class Counter(Metric):
    """Counter metric.

    Usage example:

    .. code-block:: python

        counter = Counter(
            name='frames_per_source',
            description='Number of processed frames per source',
            labelnames=('source_id',),
        )
        counter.inc(labels=('camera-1',))
    """

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
    """Gauge metric.

    Usage example:

    .. code-block:: python

        gauge = Gauge(
            name='total_queue_length',
            description='The total queue length for the pipeline',
        )
        gauge.set(123)
    """

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
