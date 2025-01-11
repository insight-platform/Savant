"""Example of how to use metrics in PyFunc."""

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.metrics import get_or_create_counter, get_or_create_gauge


def get_gauge():
    return get_or_create_gauge(
        name='total_queue_length',
        description='The total queue length for the pipeline',
        # Labels are optional, by default there are no labels
    )


def get_counter():
    return get_or_create_counter(
        name='frames_per_source',
        description='Number of processed frames per source',
        label_names=['source_id'],
    )


class PyFuncMetricsExample(NvDsPyFuncPlugin):
    """Example of how to use metrics in PyFunc.

    Metrics values example:

    .. code-block:: text

        # HELP frames_per_source_total Number of processed frames per source
        # TYPE frames_per_source_total counter
        frames_per_source_total{module_stage="tracker",source_id="city-traffic"} 748.0 1700803467794
        # HELP total_queue_length The total queue length for the pipeline
        # TYPE total_queue_length gauge
        total_queue_length{module_stage="tracker",source_id="city-traffic"} 36.0 1700803467794

    Note: the "module_stage" label is configured in docker-compose file and added to all metrics.
    """

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        # Count the frame for this source
        get_counter().inc(
            label_values=[frame_meta.source_id],
        )
        try:
            last_runtime_metric = self.get_runtime_metrics(1)[0]
            queue_length = sum(
                stage[0].queue_length for stage in last_runtime_metric.stage_stats
            )
        except IndexError:
            queue_length = 0

        # Set the total queue length for this source
        get_gauge().set(queue_length)  # The new gauge value
