OpenTelemetry Support
---------------------

**What is OpenTelemetry (from the official website).** OpenTelemetry is a collection of APIs, SDKs, and tools. Use it to instrument, generate, collect, and export telemetry data (metrics, logs, and traces) to help you analyze your software's performance and behavior.

.. tip::

    Read an introductory `article <https://blog.savant-ai.io/opentelemetry-in-savant-instrumenting-deep-learning-computer-vision-pipelines-dd42c7a65d00?source=friends_link&sk=b5a0c1d8a3554a38862f0c35007c3452>`_ on OpenTelemetry in Savant on Medium.

Why To Use OpenTelemetry
^^^^^^^^^^^^^^^^^^^^^^^^

In streaming systems, data flows through container stages in parallel what leads to messy logs with overlapping records for various messages. For example, when the message ``M[N]`` enters the pipeline, message ``M[N-P]`` passes the stage ``P``, and their logs overlap.

For a developer, navigating through those logs without advanced filtering tools is very difficult. OpenTelemetry solves the problem by introducing the concept of Trace: a unique ID corresponding to a business transaction. In Savant, every frame coming in a pipeline can have a trace id, thus making it possible to separate logs related to a specific frame.

What is more, the Trace is not a flat structure: a developer can wrap certain pieces of code with so called Spans, limiting the scope, so logs are attached to the hierarchy of spans rather than Trace directly. Every span is automatically a profiling object because OpenTelemetry collects its start time, end time and duration.

Developers can attach auxiliary information to a span: attributes, events, span statuses.

Savant automatically creates spans for every pipeline stage and gives developer API to create nested spans in their Python code.

OpenTelemetry send tracing information to a trace collector. Currently, Savant integrates with any OpenTelemetry-compatible trace collector.
.. image:: ../../../samples/telemetry/assets/01-trace.png

OpenTelemetry Sampling
^^^^^^^^^^^^^^^^^^^^^^

Sampling is an approach of statistically limiting the number of messages based on 1 of every N principle.

Depending on the sampling rate configured, the sampling fits both development/debug and production use. It allows combining code instrumenting with resource saving.

A particular case is when the sampling rate is set to 0: the pipeline module does not create traces at all but still serves externally propagated traces.

Trace Propagation
^^^^^^^^^^^^^^^^^

Trace propagation is a mechanism of passing traces between distributed, decoupled systems. Savant supports trace propagation.

OpenTelemetry Configuration (Module)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``params.telemetry.tracing`` to configure OpenTelemetry for the module.

.. warning::

    We recommend using the OpenTelemetry configuration file (JSON file path set with the ``provider_params_config`` parameter) to configure OpenTelemetry for the module. In the future versions of Savant, the module configuration can be deprecated. The motivation for this change is to simplify reuse of the OpenTelemetry configuration in different modules and adapters.

.. code-block:: yaml

    # base module parameters
    parameters:
      # DevMode (hot Python code reload on file change)
      dev_mode: True

      # enable OpenTelemetry
      telemetry:
        tracing:
          sampling_period: 100
          root_span_name: pipeline
          provider: opentelemetry
          # or (mutually exclusive with provider_params, high priority)
          # use provider config file (take a look at samples/telemetry/otlp/x509_provider_config.json)
          provider_params_config: /path/to/x509_provider_config.json
          # or (mutually exclusive with provider_params_config, low priority)
          # use provider config attributes
          provider_params:
            service_name: demo-pipeline
            # Available protocols: grpc, http_binary, http_json.
            # Protocol should be compatible with the port in endpoint.
            # See https://www.jaegertracing.io/docs/1.62/deployment/#collector
            protocol: grpc
            endpoint: "http://jaeger:4317"
            timeout: 5000 # milliseconds
            tls:
              ca: /path/to/ca.crt
              identity:
                  certificate: /path/to/client.crt
                  key: /path/to/client.key

OpenTelemetry Configuration (JSON file)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    JSON configuration files can contain environment variables. For example, ``${TRACING_SERVICE_NAME:-default-name}`` will be replaced with the value of the ``TRACING_SERVICE_NAME`` environment variable or ``default-name`` if the variable is not set.

Full example of the OpenTelemetry configuration file with TLS configuration:

.. code-block:: json

    {
      "tracer": {
          "service_name": "${TRACING_SERVICE_NAME:-default-name}",
          "protocol": "grpc",
          "endpoint": "https://jaeger:4317",
          "timeout": {
              "secs": 10,
              "nanos": 0
          },
          "tls": {
              "ca": "/opt/savant/samples/telemetry/certs/ca.crt",
              "identity": {
                  "key": "/opt/savant/samples/telemetry/certs/client.key",
                  "certificate": "/opt/savant/samples/telemetry/certs/client.crt"
              }
          }
      },
      "context_propagation_format": "w3c"
    }


An example of the OpenTelemetry configuration file without TLS configuration:

.. code-block:: json

    {
      "tracer": {
          "service_name": "${TRACING_SERVICE_NAME:-default-name}",
          "protocol": "grpc",
          "endpoint": "http://jaeger:4317",
          "timeout": {
              "secs": 10,
              "nanos": 0
          }
      },
      "context_propagation_format": "w3c"
    }


.. youtube:: DkNifuKg-kY
