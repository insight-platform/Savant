Client SDK
==========

Client SDK is a Python library enabling developing sources and sinks in a simplified way with pure Python. The SDK allows ingesting frames and their metadata to a running module and receiving the results from a running module.

The SDK is developed to solve the following needs:

- develop integration tests for Pipelines (QA);
- implement custom source adapters without deep understanding of streaming technology;
- implement custom sink adapters without deep understanding of streaming technology;
- remote development.

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
