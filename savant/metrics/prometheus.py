from typing import Any, Dict, List, Tuple

from prometheus_client import start_http_server
from prometheus_client.metrics_core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector, REGISTRY
from savant_rs.pipeline2 import (
    FrameProcessingStatRecord,
    FrameProcessingStatRecordType,
    VideoPipeline,
)

from savant.metrics.base import BaseMetricsExporter


class PrometheusMetricsExporter(BaseMetricsExporter):
    def __init__(self, pipeline: VideoPipeline, params: Dict[str, Any]):
        super().__init__(pipeline, params)
        self._port = params['port']
        self._metrics_collector = ModuleMetricsCollector()
        REGISTRY.register(self._metrics_collector)

    def start(self):
        start_http_server(self._port)
        super().start()

    def export(self, records: List[FrameProcessingStatRecord]):
        frame_based_record = None
        timestamp_based_record = None
        for record in records:
            if record.record_type == FrameProcessingStatRecordType.Frame:
                if frame_based_record is None or frame_based_record.id < record.id:
                    frame_based_record = record
            elif record.record_type == FrameProcessingStatRecordType.Timestamp:
                if (
                    timestamp_based_record is None
                    or timestamp_based_record.id < record.id
                ):
                    timestamp_based_record = record
        if frame_based_record is not None:
            self._export_single_record(frame_based_record)
        if timestamp_based_record is not None:
            self._export_single_record(timestamp_based_record)

    def _export_single_record(self, record: FrameProcessingStatRecord):
        self._logger.debug(
            'Exporting record %s (type: %s)', record.id, record.record_type
        )
        self._metrics_collector.set_metrics(record)


class ModuleMetricsCollector(Collector):
    def __init__(self):
        self._frame_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._object_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._stage_queue_length: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._stage_frame_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._stage_object_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}
        self._stage_batch_counter: Dict[Tuple[str, ...], Tuple[int, float]] = {}

    def set_metrics(self, record: FrameProcessingStatRecord):
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
        label_names = ['record_type']
        stage_label_names = ['record_type', 'stage_name']
        yield _build_metric(
            'frame_counter',
            'Number of frames passed through the module',
            label_names,
            self._frame_counter,
            CounterMetricFamily,
        )
        yield _build_metric(
            'object_counter',
            'Number of objects passed through the module',
            label_names,
            self._object_counter,
            CounterMetricFamily,
        )
        yield _build_metric(
            'stage_queue_length',
            'Queue length in the stage',
            stage_label_names,
            self._stage_queue_length,
            GaugeMetricFamily,
        )
        yield _build_metric(
            'stage_frame_counter',
            'Number of frames passed through the stage',
            stage_label_names,
            self._stage_frame_counter,
            CounterMetricFamily,
        )
        yield _build_metric(
            'stage_object_counter',
            'Number of objects passed through the stage',
            stage_label_names,
            self._stage_object_counter,
            CounterMetricFamily,
        )
        yield _build_metric(
            'stage_batch_counter',
            'Number of frame batches passed through the stage',
            stage_label_names,
            self._stage_batch_counter,
            CounterMetricFamily,
        )


def _build_metric(
    name: str,
    documentation: str,
    labels: List[str],
    values: Dict[Tuple[str, ...], Tuple[int, float]],
    metric_class,
):
    counter = metric_class(name, documentation, labels=labels)
    for labels, (value, ts) in values.items():
        counter.add_metric(labels, value, timestamp=ts)
    return counter


def _record_type_to_string(record_type: FrameProcessingStatRecordType) -> str:
    if record_type == FrameProcessingStatRecordType.Frame:
        return 'frame'
    if record_type == FrameProcessingStatRecordType.Timestamp:
        return 'timestamp'
    if record_type == FrameProcessingStatRecordType.Initial:
        return 'initial'
