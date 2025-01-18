Prometheus Metrics Support
--------------------------

Savant supports Prometheus metrics of two kinds:

* system-level metrics collected by the pipeline and adapters and provided as is;
* user-level metrics managed by the user.

The metrics are exported in the Open Metrics textual format (currently known as a Prometheus format).

The system-level metrics mostly measure queues, frames, detected objects, latencies and other valuable information related to pipeline operation.

The user-level metrics must reflect user needs.

There are two samples demonstrating metrics:

* The `Buffer Adapter <https://github.com/insight-platform/Savant/tree/develop/samples/buffer_adapter>`__ sample: demonstrates buffer-collected metrics;
* The `Pass-through Processing <https://github.com/insight-platform/Savant/tree/develop/samples/pass_through_processing>`__ sample: demonstrates adapter system-level metrics and user-level metrics.

The system-level metrics are collected according to the collection periods, configured for the pipeline with the following parameters:

.. code-block:: yaml

    parameters:
      telemetry::
        # Configure metrics
        metrics:
          # Output stats after every N frames
          frame_period: ${oc.decode:${oc.env:METRICS_FRAME_PERIOD, 10000}}
          # Output stats after every N seconds
          time_period: ${oc.decode:${oc.env:METRICS_TIME_PERIOD, null}}
          # How many last stats to keep in the memory
          history: ${oc.decode:${oc.env:METRICS_HISTORY, 100}}
          # Parameters for metrics provider
          extra_labels: ${json:${oc.env:METRICS_EXTRA_LABELS, null}}


Configuration parameters for the system-level metrics collection:

.. list-table:: Parameters
    :header-rows: 1

    * - Parameter
      - Environment variable
      - Description
      - Default
      - Example
    * - ``parameters.telemetry.metrics.frame_period``
      - ``METRICS_FRAME_PERIOD``
      - The number of frames after which the metrics are collected.
      - ``10000``
      - ``1000``
    * - ``parameters.telemetry.metrics.time_period``
      - ``METRICS_TIME_PERIOD``
      - The number of seconds after which the metrics are collected.
      - ``null``
      - ``10``
    * - ``parameters.telemetry.metrics.history``
      - ``METRICS_HISTORY``
      - The number of last stats to keep in memory. This value is required only for fetching the history with ``savant-rs`` from the pipeline runtime.
      - ``100``
      - ``10``
    * - ``parameters.telemetry.metrics.extra_labels``
      - ``METRICS_EXTRA_LABELS``
      - Additional labels for the metrics which are automatically added to every metric (both user- and system-level).
      - ``null``
      - ``{"label1": "value1", "label2": "value2"}``


User-Level Metrics
^^^^^^^^^^^^^^^^^^

User-level metrics are managed by the user and configured in ``PyFunc`` blocks with the ``savant-rs`` interface.

Savant supports two metric types:

* CounterFamily (64-bit unsigned int);
* GaugeFamily (64-bit float).

The metrics usage is demonstrated in the
`Pass-through Processing <https://github.com/insight-platform/Savant/blob/develop/samples/pass_through_processing/py_func_metrics_example.py#L9-L22>`__ sample.
Those two functions are enough in most cases; they use simplified wrappers implemented in `Savant <https://github.com/insight-platform/Savant/blob/develop/savant/metrics/__init__.py>`__ for user convenience.

Complete Metric API can be found in the ``savant-rs`` `documentation <https://insight-platform.github.io/savant-rs/modules/savant_rs/metrics.html>`__.

