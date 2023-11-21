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
        labelnames = ['stage_name']
        self._module_frame_counter: Dict[str, Counter] = {
            ns: Counter('module_frame_counter', 'TODO', namespace=ns)
            for ns in namespaces
        }
        self._module_object_counter: Dict[str, Counter] = {
            ns: Counter('module_object_counter', 'TODO', namespace=ns)
            for ns in namespaces
        }
        self._queue_length: Dict[str, Gauge] = {
            ns: Gauge('queue_length', 'TODO', labelnames=labelnames, namespace=ns)
            for ns in namespaces
        }
        self._frame_counter: Dict[str, Counter] = {
            ns: Counter('frame_counter', 'TODO', labelnames=labelnames, namespace=ns)
            for ns in namespaces
        }
        self._object_counter: Dict[str, Counter] = {
            ns: Counter('object_counter', 'TODO', labelnames=labelnames, namespace=ns)
            for ns in namespaces
        }
        self._batch_counter: Dict[str, Counter] = {
            ns: Counter('batch_counter', 'TODO', labelnames=labelnames, namespace=ns)
            for ns in namespaces
        }

    def start(self):
        start_http_server(self._port)
        super().start()

    def export(self, records: List[FrameProcessingStatRecord]):
        for record in records:
            namespace = _get_metric_namespace(record.record_type)
            if namespace is None:
                continue
            _update_counter(self._module_frame_counter[namespace], record.frame_no)
            _update_counter(
                self._module_object_counter[namespace], record.object_counter
            )
            for stage in record.stage_stats:
                self._queue_length[namespace].labels(stage.stage_name).set(
                    stage.queue_length
                )
                _update_counter(
                    self._frame_counter[namespace].labels(stage.stage_name),
                    stage.frame_counter,
                )
                _update_counter(
                    self._object_counter[namespace].labels(stage.stage_name),
                    stage.object_counter,
                )
                _update_counter(
                    self._batch_counter[namespace].labels(stage.stage_name),
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
