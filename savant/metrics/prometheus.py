from typing import Any, Dict, List

from savant_rs.metrics import set_extra_labels
from savant_rs.webserver import *

from savant.metrics.base import BaseMetricsExporter
from savant.utils.logging import get_logger

logger = get_logger(__name__)


class PrometheusMetricsExporter(BaseMetricsExporter):
    """Prometheus metrics exporters.

    :param params: provider parameters
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self._port = params['port'] or 8888
        extra_labels = params.get('labels') or {}
        set_extra_labels(extra_labels)

    def start(self):
        logger.debug('Starting Prometheus metrics exporter on port %s', self._port)
        init_webserver(self._port)
        logger.debug('Registering metrics collector')
        logger.info('Started Prometheus metrics exporter on port %s', self._port)

    def stop(self):
        logger.debug('Unregistering metrics collector')
        stop_webserver()
