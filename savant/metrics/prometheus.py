from typing import Any, Dict, List

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
        label_names = ['record_type']
        stage_label_names = ['record_type', 'stage_name']

        self._frame_counter = Counter(
            'frame_counter',
            'Number of frames passed through the module',
            labelnames=label_names,
        )
        self._object_counter = Counter(
            'object_counter',
            'Number of objects passed through the module',
            labelnames=label_names,
        )

        self._stage_queue_length = Gauge(
            'stage_queue_length',
            'Queue length in the stage',
            labelnames=stage_label_names,
        )
        self._stage_frame_counter = Counter(
            'stage_frame_counter',
            'Number of frames passed through the stage',
            labelnames=stage_label_names,
        )
        self._stage_object_counter = Counter(
            'stage_object_counter',
            'Number of objects passed through the stage',
            labelnames=stage_label_names,
        )
        self._stage_batch_counter = Counter(
            'stage_batch_counter',
            'Number of frame batches passed through the stage',
            labelnames=stage_label_names,
        )

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
        record_type_str = _record_type_to_string(record.record_type)
        _update_counter(
            self._frame_counter.labels(record_type_str),
            record.frame_no,
        )
        _update_counter(
            self._object_counter.labels(record_type_str),
            record.object_counter,
        )
        for stage in record.stage_stats:
            self._stage_queue_length.labels(record_type_str, stage.stage_name).set(
                stage.queue_length
            )
            _update_counter(
                self._stage_frame_counter.labels(record_type_str, stage.stage_name),
                stage.frame_counter,
            )
            _update_counter(
                self._stage_object_counter.labels(record_type_str, stage.stage_name),
                stage.object_counter,
            )
            _update_counter(
                self._stage_batch_counter.labels(record_type_str, stage.stage_name),
                stage.batch_counter,
            )


def _record_type_to_string(record_type: FrameProcessingStatRecordType) -> str:
    if record_type == FrameProcessingStatRecordType.Frame:
        return 'frame'
    if record_type == FrameProcessingStatRecordType.Timestamp:
        return 'timestamp'
    if record_type == FrameProcessingStatRecordType.Initial:
        return 'initial'


def _update_counter(counter: Counter, value: int):
    last_value = list(counter.collect())[0].samples[0].value
    counter.inc(value - last_value)
