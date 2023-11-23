from typing import Any, Dict, List

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.metrics_core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector
from savant_rs.pipeline2 import (
    FrameProcessingStatRecord,
    FrameProcessingStatRecordType,
    VideoPipeline,
)

from savant.metrics.base import BaseMetricsExporter, Counter, Gauge, Metric
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
        self._registry = CollectorRegistry()
        self._metrics_collector = ModuleMetricsCollector(pipeline, labels)

    def start(self):
        logger.debug('Starting Prometheus metrics exporter on port %s', self._port)
        start_http_server(self._port, registry=self._registry)
        logger.debug('Registering metrics collector')
        self._registry.register(self._metrics_collector)
        logger.info('Started Prometheus metrics exporter on port %s', self._port)

    def stop(self):
        logger.debug('Unregistering metrics collector')
        self._registry.unregister(self._metrics_collector)
        # TODO: stop the server

    def register_metric(self, metric: Metric):
        logger.debug('Registering metric %s of type %s', metric.name, type(metric))
        self._metrics_collector.register_metric(metric)


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
        self._extra_label_names = tuple(name for name, _ in extra_labels)
        self._extra_label_values = tuple(value for _, value in extra_labels)

        label_names = ('record_type',)
        stage_label_names = ('record_type', 'stage_name')
        self._metrics = {
            'frame_counter': Counter(
                'frame_counter',
                'Number of frames passed through the module',
                label_names,
            ),
            'object_counter': Counter(
                'object_counter',
                'Number of objects passed through the module',
                label_names,
            ),
            'stage_queue_length': Gauge(
                'stage_queue_length',
                'Queue length in the stage',
                stage_label_names,
            ),
            'stage_frame_counter': Counter(
                'stage_frame_counter',
                'Number of frames passed through the stage',
                stage_label_names,
            ),
            'stage_object_counter': Counter(
                'stage_object_counter',
                'Number of objects passed through the stage',
                stage_label_names,
            ),
            'stage_batch_counter': Counter(
                'stage_batch_counter',
                'Number of frame batches passed through the stage',
                stage_label_names,
            ),
        }

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
        self._metrics['frame_counter'].set(record.frame_no, labels, ts)
        self._metrics['object_counter'].set(record.object_counter, labels, ts)
        for stage in record.stage_stats:
            stage_labels = record_type_str, stage.stage_name
            self._metrics['stage_queue_length'].set(
                stage.queue_length, stage_labels, ts
            )
            self._metrics['stage_frame_counter'].set(
                stage.frame_counter, stage_labels, ts
            )
            self._metrics['stage_object_counter'].set(
                stage.object_counter, stage_labels, ts
            )
            self._metrics['stage_batch_counter'].set(
                stage.batch_counter, stage_labels, ts
            )

    def collect(self):
        """Build and collect all metrics."""

        logger.trace('Collecting metrics')
        for record in self.get_last_records():
            self.update_metrics(record)
        yield from self.build_all_metrics()

    def build_all_metrics(self):
        """Build all metrics for Promethus to collect."""

        logger.trace('Building metrics')
        for metric in self._metrics.values():
            yield self.build_metric(metric)

    def build_metric(self, metric: Metric):
        """Build metric for Prometheus to collect."""

        logger.trace('Building metric %s', metric.name)
        if isinstance(metric, Counter):
            metric_class = CounterMetricFamily
        elif isinstance(metric, Gauge):
            metric_class = GaugeMetricFamily
        else:
            raise ValueError(
                f'Failed to build metric {metric.name}: unsupported metric type {type(metric)}'
            )
        prom_metric = metric_class(
            name=metric.name,
            documentation=metric.description,
            labels=metric.labelnames + self._extra_label_names,
        )
        for labels, (value, ts) in metric.values.items():
            logger.trace('Building metric %s for labels %s', metric.name, labels)
            prom_metric.add_metric(
                labels=labels + self._extra_label_values,
                value=value,
                timestamp=ts,
            )

        return prom_metric

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

    def register_metric(self, metric: Metric):
        if not isinstance(metric, (Counter, Gauge)):
            raise ValueError(
                f'Failed to register metric {metric.name}: unsupported metric type {type(metric)}'
            )
        if metric.name in self._metrics:
            raise ValueError(
                f'Failed to register metric {metric.name}: metric already exists'
            )
        self._metrics[metric.name] = metric


def _record_type_to_string(record_type: FrameProcessingStatRecordType) -> str:
    # Cannot use dict since FrameProcessingStatRecordType is not hashable
    if record_type == FrameProcessingStatRecordType.Frame:
        return 'frame'
    if record_type == FrameProcessingStatRecordType.Timestamp:
        return 'timestamp'
    if record_type == FrameProcessingStatRecordType.Initial:
        return 'initial'
