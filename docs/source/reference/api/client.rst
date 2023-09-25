Client SDK
==========

Client SDK is a Python library enabling developing sources and sinks in a simplified way with pure Python. The SDK allows ingesting frames and their metadata to a running module and receiving the results from a running module.

The SDK is developed to solve the following needs:

- develop integration tests for Pipelines (QA);
- implement custom source adapters without deep understanding of streaming technology;
- implement custom sink adapters without deep understanding of streaming technology;
- remote development.

Source usage example:

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
        # Note: healthcheck port should be configured in the module.
        .with_module_health_check_url('http://172.17.0.1:8888/healthcheck')
        .build()
    )

    # Send a JPEG image from a file to the module
    result = source(JpegSource('cam-1', 'data/AVG-TownCentre.jpeg'))
    print(result.status)
    time.sleep(1)  # Wait for the module to process the frame
    result.logs().pretty_print()

Sink usage example:

.. code-block:: python

    from savant.client import JaegerLogProvider, SinkBuilder

    # Build the sink
    sink = (
        SinkBuilder()
        .with_socket('sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc')
        .with_idle_timeout(60)
        .with_log_provider(JaegerLogProvider('http://localhost:16686'))
        # Note: healthcheck port should be configured in the module.
        .with_module_health_check_url('http://172.17.0.1:8888/healthcheck')
        .build()
    )

    # Receive results from the module and print them
    for result in sink:
        print(result.frame_meta)
        result.logs().pretty_print()

Async example (both source and sink):

.. code-block:: python

    import asyncio
    from savant_rs import init_jaeger_tracer
    from savant.client import JaegerLogProvider, JpegSource, SinkBuilder, SourceBuilder


    async def run_source():
        # Build the source
        source = (
            SourceBuilder()
            .with_log_provider(JaegerLogProvider('http://localhost:16686'))
            .with_socket('pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc')
            .build_async()
        )

        # Send a JPEG image from a file to the module
        result = await source(JpegSource('cam-1', 'data/AVG-TownCentre.jpeg'))
        print(result.status)


    async def run_sink():
        # Build the sink
        sink = (
            SinkBuilder()
            .with_socket('sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc')
            .with_idle_timeout(60)
            .with_log_provider(JaegerLogProvider('http://localhost:16686'))
            .build_async()
        )

        # Receive results from the module and print them
        async for result in sink:
            print(result.frame_meta)
            result.logs().pretty_print()


    async def main():
        # Initialize Jaeger tracer to send metrics and logs to Jaeger.
        # Note: the Jaeger tracer also should be configured in the module.
        init_jaeger_tracer('savant-client', 'localhost:6831')
        await asyncio.gather(run_sink(), run_source())


    asyncio.run(main())

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
    runner.source.AsyncSourceRunner
    runner.sink.SinkRunner
    runner.sink.AsyncSinkRunner
