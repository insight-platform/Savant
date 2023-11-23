from typing import Any, Dict, List, Tuple

from prometheus_client import start_http_server
from prometheus_client.metrics_core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import REGISTRY, Collector
from savant_rs.pipeline2 import (
    FrameProcessingStatRecord,
    FrameProcessingStatRecordType,
    VideoPipeline,
)

from savant.metrics.base import BaseMetricsExporter
from savant.utils.logging import get_logger

logger = get_logger(__name__)


class PrometheusMetricsExporter(BaseMetricsExporter):
    """Prometheus metrics exporters.

    :param pipeline: VideoPipeline instance
    :param params: provider parameters
    """

    def __init__(self, pipeline: VideoPipeline, params: Dict[str, Any]):
        self._port = params['port']
        labels = params.get('labels') or {}
        self._metrics_collector = ModuleMetricsCollector(pipeline, labels)

    def start(self):
        logger.debug('Starting Prometheus metrics exporter on port %s', self._port)
        start_http_server(self._port)
        logger.debug('Registering metrics collector')
        REGISTRY.register(self._metrics_collector)
        logger.info('Started Prometheus metrics exporter on port %s', self._port)

    def stop(self):
        logger.debug('Unregistering metrics collector')
        REGISTRY.unregister(self._metrics_collector)
        # TODO: stop the server


class ModuleMetricsCollector(Collector):
    """Collector for module metrics with timestamps."""

    def __init__(
        self,
        pipeline: VideoPipeline,
        extra_labels: Dict[str, str],
    ):
        self._last_record_id = -1
        self._pipeline = pipeline

        extra_labels = sorted(extra_labels.items())
        extra_label_names = [name for name, _ in extra_labels]
        self._label_names = ['record_type'] + extra_label_names
        self._stage_label_names = ['record_type', 'stage_name'] + extra_label_names
        self._extra_label_values = tuple(value for _, value in extra_labels)

        self._frame_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._object_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._stage_queue_length: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._stage_frame_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._stage_object_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._stage_batch_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}

    def update_metrics(self, record: FrameProcessingStatRecord):
        """Update metrics values."""

        logger.debug(
            'Updating metrics with record %s of type %s. Timestamp: %s ms.',
            record.id,
            record.record_type,
            record.ts,
        )
        record_type_str = _record_type_to_string(record.record_type)
        ts = record.ts / 1000
        labels = (record_type_str,)
        self._frame_counter[labels] = record.frame_no, ts
        self._object_counter[labels] = record.object_counter, ts
        for stage in record.stage_stats:
            stage_labels = record_type_str, stage.stage_name
            self._stage_queue_length[stage_labels] = (stage.queue_length, ts)
            self._stage_frame_counter[stage_labels] = (stage.frame_counter, ts)
            self._stage_object_counter[stage_labels] = (stage.object_counter, ts)
            self._stage_batch_counter[stage_labels] = (stage.batch_counter, ts)

    def collect(self):
        """Build and collect all metrics."""

        logger.trace('Collecting metrics')
        for record in self.get_last_records():
            self.update_metrics(record)
        yield from self.build_all_metrics()

    def build_all_metrics(self):
        """Build all metrics for Promethus to collect."""

        logger.trace('Building metrics')
        yield self.build_metric(
            'frame_counter',
            'Number of frames passed through the module',
            self._label_names,
            self._frame_counter,
            CounterMetricFamily,
        )
        yield self.build_metric(
            'object_counter',
            'Number of objects passed through the module',
            self._label_names,
            self._object_counter,
            CounterMetricFamily,
        )
        yield self.build_metric(
            'stage_queue_length',
            'Queue length in the stage',
            self._stage_label_names,
            self._stage_queue_length,
            GaugeMetricFamily,
        )
        yield self.build_metric(
            'stage_frame_counter',
            'Number of frames passed through the stage',
            self._stage_label_names,
            self._stage_frame_counter,
            CounterMetricFamily,
        )
        yield self.build_metric(
            'stage_object_counter',
            'Number of objects passed through the stage',
            self._stage_label_names,
            self._stage_object_counter,
            CounterMetricFamily,
        )
        yield self.build_metric(
            'stage_batch_counter',
            'Number of frame batches passed through the stage',
            self._stage_label_names,
            self._stage_batch_counter,
            CounterMetricFamily,
        )

    def build_metric(
        self,
        name: str,
        documentation: str,
        label_names: List[str],
        values: Dict[Tuple[str, ...], Tuple[int, float]],
        metric_class,
    ):
        """Build metric for Prometheus to collect."""

        logger.trace('Building metric %s', name)
        counter = metric_class(name, documentation, labels=label_names)
        for labels, (value, ts) in values.items():
            counter.add_metric(labels + self._extra_label_values, value, timestamp=ts)

        return counter

    def get_last_records(self) -> List[FrameProcessingStatRecord]:
        """Get last metrics records from the pipeline.

        :returns: list of FrameProcessingStatRecord: last frame-based and
                  timestamp-based records.
        """

        last_record_id = self._last_record_id
        frame_based_record = None
        timestamp_based_record = None

        for record in self._pipeline.get_stat_records(100):  # TODO: use last_record_id
            if record.id <= self._last_record_id:
                continue
            if record.record_type == FrameProcessingStatRecordType.Frame:
                if frame_based_record is None or frame_based_record.id < record.id:
                    frame_based_record = record
            elif record.record_type == FrameProcessingStatRecordType.Timestamp:
                if (
                    timestamp_based_record is None
                    or timestamp_based_record.id < record.id
                ):
                    timestamp_based_record = record
            last_record_id = max(last_record_id, record.id)
        self._last_record_id = last_record_id
        records = []
        if frame_based_record is not None:
            records.append(frame_based_record)
        if timestamp_based_record is not None:
            records.append(timestamp_based_record)

        return records


def _record_type_to_string(record_type: FrameProcessingStatRecordType) -> str:
    # Cannot use dict since FrameProcessingStatRecordType is not hashable
    if record_type == FrameProcessingStatRecordType.Frame:
        return 'frame'
    if record_type == FrameProcessingStatRecordType.Timestamp:
        return 'timestamp'
    if record_type == FrameProcessingStatRecordType.Initial:
        return 'initial'
