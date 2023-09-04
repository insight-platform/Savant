Client SDK
==========

Client SDK is a Python library enabling developing sources and sinks in a simplified way with pure Python. The SDK allows ingesting frames and their metadata to a running module and receiving the results from a running module.

The SDK is developed to solve the following needs:

- develop integration tests for Pipelines (QA);
- implement custom source adapters without deep understanding of streaming technology;
- implement custom sink adapters without deep understanding of streaming technology;
- remote development.

OpenTelemetry in Client SDK
---------------------------

Savant supports OpenTelemetry for collecting traces and logs. Frames, delivered to a module, can optionally include OpenTelemetry propagated information, which allows instrumenting pipelines precisely.

.. note::

    In practice, we recommend using sampled OpenTelemetry traces; however, Client SDK creates a trace for every single frame sent to the module.

Client SDK integrates with OpenTelemetry, which means that every time a frame is sent to a module, it contains a new trace, thus the trace causes the module to instrument its internal parts, creating necessary OpenTelemetry spans in the pipeline, including pyfuncs. The developer can create auxiliary spans within her code to find out what happens. Please, refer to :doc:`/advanced_topics/9_open_telemetry` for details.

During the frame processing ongoing logs are attached to the currently active span and thus collected and associated with a specific trace. The developer also can attach additional information to the trace by calling corresponding span `methods <https://insight-platform.github.io/savant-rs/modules/savant_rs/utils.html#savant_rs.utils.TelemetrySpan>`_.

When the frame is sent, the :py:class:`runner.source.SourceResult` is returned. The developer can retrieve the `trace_id` from it for matching sent and delivered frames.

.. note::

    The `trace_id` can also be used to observe the trace in the Opentelemetry management system like Jaeger.

When the frame processing result is retrieved from the module, the developer can request the frame, metadata and logs collected by OpenTelemetry.

Currently we support only Jaeger OpenTelemetry collector. Logs are fetched from Jaeger REST API.

Remote Development
------------------

Client SDK enables remote development, allowing running programmatic sources and sinks locally, while processing the data remotely. It can be convenient in the  following cases:

- develop on a host machine without GPU;
- develop on remote hardware with no KVM access (e.g. Jetson or dGPU in a datacenter).

To utilize the full power of Client SDK it must be paired with:

- :doc:`/advanced_topics/9_open_telemetry`
- :doc:`/advanced_topics/9_dev_server`

To find out more, explore :doc:`/getting_started/2_module_devguide`.

Source Example
--------------

Sources ingest frames and their metadata to a running module.

.. note::

    Currently, Client SDK supports only JPEG source, but you can implement your own source based on :py:class:`JpegSource`.

.. code-block:: python

    import time
    from savant_rs import init_jaeger_tracer
    from savant.client import JaegerLogProvider, JpegSource, SourceBuilder

    # Initialize Jaeger tracer to send metrics and logs to Jaeger.
    # Note: the Jaeger tracer also should be configured in the module.
    init_jaeger_tracer('savant-client', 'localhost:6831')

    # Build the source
    source = (
        SourceBuilder()
        .with_log_provider(JaegerLogProvider('http://localhost:16686'))
        .with_socket('pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc')
        .build()
    )

    # Send a JPEG image from a file to the module
    result = source(JpegSource('cam-1', 'data/AVG-TownCentre.jpeg'))
    print(result.status)
    time.sleep(1)  # Wait for the module to process the frame
    result.logs().pretty_print()


Sink Example
------------

Sinks retrieve results from a module.

.. code-block:: python

    from savant.client import JaegerLogProvider, SinkBuilder

    # Build the sink
    sink = (
        SinkBuilder()
        .with_socket('sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc')
        .with_idle_timeout(60)
        .with_log_provider(JaegerLogProvider('http://localhost:16686'))
        .build()
    )

    # Receive results from the module and print them
    for result in sink:
        print(result.frame_meta)
        result.logs().pretty_print()

.. currentmodule:: savant.client

Builders
--------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    SourceBuilder
    SinkBuilder

Frame sources
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    FrameSource
    JpegSource

Log providers
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    LogProvider
    JaegerLogProvider

Results
-------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    runner.source.SourceResult
    runner.sink.SinkResult

Runners
-------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: autosummary/class.rst

    runner.source.SourceRunner
    runner.sink.SinkRunner
