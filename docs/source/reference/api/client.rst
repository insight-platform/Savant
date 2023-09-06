Client
======

Source/Sink framework for development and QA purposes. You can send frames to the module and receive results with this framework.

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
        .build()
    )

    # Send a JPEG image from a file to the module
    result = source(JpegSource('cam-1', 'data/AVG-TownCentre.jpeg'))
    print(result.status)
    time.sleep(1)  # Wait for the module to process the frame
    result.logs().pretty_print()

Source usage example (async):

.. code-block:: python

    import asyncio
    import time
    from savant_rs import init_jaeger_tracer
    from savant.client import JaegerLogProvider, JpegSource, SourceBuilder

    async def main():
        # Initialize Jaeger tracer to send metrics and logs to Jaeger.
        # Note: the Jaeger tracer also should be configured in the module.
        init_jaeger_tracer('savant-client', 'localhost:6831')

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
        time.sleep(1)  # Wait for the module to process the frame
        result.logs().pretty_print()

    asyncio.run(main())

Sink usage example:

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
    runner.source.AsyncSourceRunner
    runner.sink.SinkRunner
