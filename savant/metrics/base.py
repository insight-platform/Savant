import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import Any, Dict, List

from savant_rs.pipeline2 import FrameProcessingStatRecord, VideoPipeline

from savant.utils.logging import get_logger

DEFAULT_EXPORT_INTERVAL = 1000


class BaseMetricsExporter(ABC):
    """Base class for metrics exporters.

    :param pipeline: VideoPipeline instance
    :param params: provider parameters
    """

    _thread: Thread

    def __init__(self, pipeline: VideoPipeline, params: Dict[str, Any]):
        self._logger = get_logger(
            f'{self.__class__.__module__}.{self.__class__.__name__}'
        )
        self._pipeline = pipeline
        self._export_interval = (
            params.get('export_interval', DEFAULT_EXPORT_INTERVAL) / 1000
        )
        self._is_running = False
        self._last_record_id = -1

    def start(self):
        """Start metrics exporter."""

        self._is_running = True
        self._thread = Thread(target=self.run, daemon=True)
        self._thread.start()
        self._logger.info('Metrics exporter started')

    def stop(self):
        """Stop metrics exporter."""

        self._is_running = False
        self._thread.join(timeout=self._export_interval)
        self._logger.info('Metrics exporter stopped')

    def run(self):
        """Run metrics exporter in a loop."""

        while self._is_running:
            time.sleep(self._export_interval)
            try:
                self._export_last_records()
            except Exception as e:
                self._logger.error('Failed to export metrics: %s', e)

    def _export_last_records(self):
        next_last_record = -1
        records = {}
        record: FrameProcessingStatRecord
        for record in self._pipeline.get_stat_records(100):  # TODO: use last_record_id
            if record.id <= self._last_record_id:
                continue
            if records.setdefault(str(record.record_type), record).id < record.id:
                records[str(record.record_type)] = record
            next_last_record = max(next_last_record, record.id)
        if not records:
            self._logger.trace('No records to export')
            return

        records = list(records.values())
        self._logger.debug(
            'Exporting %d records. Last record ID is %s.',
            len(records),
            next_last_record,
        )
        self.export(records)
        self._last_record_id = next_last_record

    @abstractmethod
    def export(self, records: List[FrameProcessingStatRecord]):
        """Export metrics."""
        pass
