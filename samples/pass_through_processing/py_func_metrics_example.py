"""Example of how to use metrics in PyFunc."""

from savant.deepstream.meta.frame import NvDsFrameMeta
from savant.deepstream.pyfunc import NvDsPyFuncPlugin
from savant.gstreamer import Gst
from savant.metrics import Counter, Gauge


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

    # Called when the new source is added
    def on_source_add(self, source_id: str):
        # Check if the metric is not registered yet
        if 'frames_per_source' not in self.metrics:
            # Register the counter metric
            self.metrics['frames_per_source'] = Counter(
                name='frames_per_source',
                description='Number of processed frames per source',
                # Labels are optional, by default there are no labels
                labelnames=('source_id',),
            )
            self.logger.info('Registered metric: %s', 'frames_per_source')
        if 'total_queue_length' not in self.metrics:
            # Register the gauge metric
            self.metrics['total_queue_length'] = Gauge(
                name='total_queue_length',
                description='The total queue length for the pipeline',
                # There are no labels for this metric
            )
            self.logger.info('Registered metric: %s', 'total_queue_length')

    def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
        # Count the frame for this source
        self.metrics['frames_per_source'].inc(
            # 1,  # Default increment value
            # Labels should be a tuple and must match the labelnames
            labels=(frame_meta.source_id,),
        )
        try:
            last_runtime_metric = self.get_runtime_metrics(1)[0]
            queue_length = sum(
                stage.queue_length for stage in last_runtime_metric.stage_stats
            )
        except IndexError:
            queue_length = 0

        # Set the total queue length for this source
        self.metrics['total_queue_length'].set(
            queue_length,  # The new gauge value
            # There are no labels for this metric
        )
