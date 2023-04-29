Configuration Variables Interpolation
=====================================

When initializing a module with the configuration file, Savant uses the `OmegaConf <https://github.com/omry/omegaconf>`__ package enabling not only immediate parameter values, but also references to the values of other parameters, as well as options for interpolating configuration parameters from external sources.

Interpolation is useful in the following cases:

* configure an option to set the value from an environment variable, with a default value given;
* avoid duplication of the same value in several places, for example, in the case when objects of the same class are filtered at the output of the Detection Unit and processed in the Python Function Unit;
* use for the parameter a derivative of the value of another parameter if the values of these parameters are interrelated; for example, when the minimum allowed sizes for detected objects depend on the frame size.

OmegaConf's capabilities to interpolate configuration values are described in OmegaConf `interpolation <https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation>`__ manual. Savant allows using any of the features of OmegaConf, for example:

.. code-block:: yaml

    parameters:
      object_min_size: 64
    pipeline:
      elements:
        - element: nvinfer@detector
          # irrelevant configuration is omitted
          output:
            # irrelevant configuration is omitted
            objects:
              - class_id: 0
                label: person
                selector:
                  kwargs:
                    min_width: ${parameters.object_min_size}
                    min_height: ${.min_width}

There are resolvers available that can be used to resolve variables from external environments, such as a combination of ``oc.env`` and ``oc.decode`` to load a parameter value from the operating system environment:

.. code-block:: yaml

    parameters:
      frame:
        width: ${oc.decode:${oc.env:FRAME_WIDTH, 1280}}
        height: ${oc.decode:${oc.env:FRAME_HEIGHT, 720}}

In addition, Savant implements special resolvers.

Initializer Resolver
--------------------

The ``initializer`` resolver makes it easy to get parameter values from any supported source (environment, etcd, default). When using the ``initializer``, the resolution for value happens according to the priorities configured with the ``parameter_init_priority`` section. By default, the priorities are configured as follows:

.. code-block:: yaml

    # Etcd -> env -> default
    #
    parameter_init_priority:
      # higher number means lower priority
      environment: 20
      etcd: 10
      # default has the lowest priority

To use the ``initializer`` resolver, you need to specify the type name of the resolver ``initializer``, a name of the parameter by which the value will be searched among the sources, and a default value. For example, using the ``initializer`` looks like this:

.. code-block:: yaml

    parameters:
      fps_period: ${initializer:FPS_PERIOD, 10000}
      # equivalent omega conf resolvers
      # fps_period: ${oc.decode:${oc.env:FPS_PERIOD, 10000}}

Calc Resolver
-------------

The ``calc`` resolver allows using arithmetic expressions to calculate the value of a configuration parameter, including based on the value of other configuration parameters. For example, the configuration that sets the filtering of detected objects according to the frame size in the pipeline looks like this:

.. code-block:: yaml

    parameters:
      frame:
        width: 1280
        height: 720
    pipeline:
      elements:
        - element: nvinfer@detector
          # skip irrelevant configuration
          output:
          # skip irrelevant configuration
            objects:
              - class_id: 0
                label: person
                selector:
                  kwargs:
                    min_width: ${calc:"arg_0*arg_1", ${parameters.frame.width}, 0.15}
                    min_height: ${calc:"arg_0*arg_1", ${parameters.frame.height}, 0.15}

The structure of the ``calc`` resolver from the example above is:

* the ``calc`` specifies resolver name;
* ``arg_0 * arg_1`` an arithmetic expression to evaluate the value;
* then, separated by commas, the arguments for the arithmetic expression are listed, which can be specified both directly and using interpolation, while the first of the arguments will be substituted for ``arg_0``, the second will be substituted for ``arg_1`` and so on, the number the arguments passed must match the number of names of the form ``arg_x`` in the string of the arithmetic expression.

The ``calc`` resolver relies on the ``simpleeval`` package, a list of supported operators can be seen at the `Operators <https://github.com/danthedeckie/simpleeval#operators>`__ page.

JSON Resolver
-------------

The JSON resolver decodes a JSON string, resulting in a JSON object that will be substituted into the YAML-configuration under the node to which the resolver belongs.

Such a resolver is useful if you want to pass a whole section through an environment variable or Etcd. For example, the default value of the ``parameters.output_frame`` parameter is internally defined as follows:

.. code-block:: yaml

    parameters:
      output_frame: ${json:${oc.env:OUTPUT_FRAME, null}}

As a result, when loading a configuration, Savant tries to fetch the value of the ``OUTPUT_FRAME`` environment variable, and then decodes the resulting string as JSON. That is, by passing the following line to the module configuration through the ``OUTPUT_FRAME`` environment variable:

.. code-block:: text

    '{"codec": "h264", "encoder_params": {"bitrate": 4000000}}'

The result will be the following configuration:

.. code-block:: yaml

    parameters:
      output_frame:
        codec: h264
        encoder_params:
          bitrate: 4000000

