from typing import Optional

from savant_rs.pipeline2 import VideoPipeline

from savant.config.schema import MetricsParameters
from savant.metrics.base import BaseMetricsExporter
from savant.metrics.prometheus import ModuleMetricsCollector, PrometheusMetricsExporter


def build_metrics_exporter(
    pipeline: VideoPipeline,
    params: MetricsParameters,
) -> Optional[BaseMetricsExporter]:
    """Build metrics exporter."""

    if params.provider is None:
        return None

    if params.provider == 'prometheus':
        return PrometheusMetricsExporter(
            params.provider_params,
            ModuleMetricsCollector(
                pipeline,
                params.provider_params.get('labels') or {},
            ),
        )

    raise ValueError(f'Unknown metrics provider: {params.provider}')
