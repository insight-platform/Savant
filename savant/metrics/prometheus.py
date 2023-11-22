from typing import Any, Dict, List, Optional

from prometheus_client import Counter, Gauge, start_http_server
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
        namespaces = [
            _get_metric_namespace(FrameProcessingStatRecordType.Frame),
            _get_metric_namespace(FrameProcessingStatRecordType.Timestamp),
        ]

        self._frame_counter: Dict[str, Counter] = {
            ns: Counter(
                'frame_counter',
                'Number of frames passed through the module',
                namespace=ns,
            )
            for ns in namespaces
        }
        self._object_counter: Dict[str, Counter] = {
            ns: Counter(
                'object_counter',
                'Number of objects passed through the module',
                namespace=ns,
            )
            for ns in namespaces
        }

        labelnames = ['stage_name']
        self._stage_queue_length: Dict[str, Gauge] = {
            ns: Gauge(
                'stage_queue_length',
                'Queue length in the stage',
                labelnames=labelnames,
                namespace=ns,
            )
            for ns in namespaces
        }
        self._stage_frame_counter: Dict[str, Counter] = {
            ns: Counter(
                'stage_frame_counter',
                'Number of frames passed through the stage',
                labelnames=labelnames,
                namespace=ns,
            )
            for ns in namespaces
        }
        self._stage_object_counter: Dict[str, Counter] = {
            ns: Counter(
                'stage_object_counter',
                'Number of objects passed through the stage',
                labelnames=labelnames,
                namespace=ns,
            )
            for ns in namespaces
        }
        self._stage_batch_counter: Dict[str, Counter] = {
            ns: Counter(
                'stage_batch_counter',
                'Number of frame batches passed through the stage',
                labelnames=labelnames,
                namespace=ns,
            )
            for ns in namespaces
        }

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
        namespace = _get_metric_namespace(record.record_type)
        _update_counter(self._frame_counter[namespace], record.frame_no)
        _update_counter(self._object_counter[namespace], record.object_counter)
        for stage in record.stage_stats:
            self._stage_queue_length[namespace].labels(stage.stage_name).set(
                stage.queue_length
            )
            _update_counter(
                self._stage_frame_counter[namespace].labels(stage.stage_name),
                stage.frame_counter,
            )
            _update_counter(
                self._stage_object_counter[namespace].labels(stage.stage_name),
                stage.object_counter,
            )
            _update_counter(
                self._stage_batch_counter[namespace].labels(stage.stage_name),
                stage.batch_counter,
            )


def _get_metric_namespace(record_type: FrameProcessingStatRecordType) -> Optional[str]:
    if record_type == FrameProcessingStatRecordType.Frame:
        return 'frame_based'
    if record_type == FrameProcessingStatRecordType.Timestamp:
        return 'time_based'


def _update_counter(counter: Counter, value: int):
    last_value = list(counter.collect())[0].samples[0].value
    counter.inc(value - last_value)
