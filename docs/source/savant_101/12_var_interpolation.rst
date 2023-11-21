Module Variables
================

When initializing a module with the configuration file, Savant uses the `OmegaConf <https://github.com/omry/omegaconf>`__ package enabling not only immediate parameter values but also references to the values of other parameters, as well as options for interpolating configuration parameters from external sources.

Interpolation is useful in the following cases:

* configure an option to set the value from an environment variable, with a default value given;
* avoid duplication of the same value in several places, for example, in the case when objects of the same class are filtered at the output of the Detection Unit and also processed in the Python Function Unit;
* use for the parameter a derivative of the value of another parameter if the values of these parameters are interrelated; for example, when the minimum allowed sizes for detected objects depend on the frame size.

.. note:: Interpolation happens only **once**, when the module is being initialized.

OmegaConf value interpolation capabilities are described in OmegaConf `interpolation <https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation>`__ manual. Savant supports all features of OmegaConf, for example:

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

Environment Resolver
--------------------

In Savant, there is a resolver used to retrieve variables from the system environment, such as a combination of ``oc.env`` and ``oc.decode`` to load a parameter value from the operating system environment:

.. code-block:: yaml

    parameters:
      frame:
        width: ${oc.decode:${oc.env:FRAME_WIDTH, 1280}}
        height: ${oc.decode:${oc.env:FRAME_HEIGHT, 720}}

In addition, Savant implements special resolvers.

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

The explanation of the ``calc`` resolver usage from the example above is as follows:

* ``calc`` specifies resolver name;
* ``arg_0 * arg_1`` is the arithmetic expression to evaluate the value;
* the list of values corresponding to the arguments of the expression, separated by commas.

The ``calc`` resolver relies on the ``simpleeval`` package, a list of supported operators can be seen at the `Operators <https://github.com/danthedeckie/simpleeval#operators>`__ page.

JSON Resolver
-------------

The JSON resolver decodes a JSON string, resulting in the JSON object that will be placed into the YAML configuration under the node to which the expression belongs.

Such a resolver is useful if you want to pass a whole section through an environment variable or Etcd. For example, the default value of the ``parameters.output_frame`` parameter is internally defined as follows:

.. code-block:: yaml

    parameters:
      output_frame: ${json:${oc.env:OUTPUT_FRAME, null}}

As a result, when loading a configuration, Savant tries to fetch the value of the ``OUTPUT_FRAME`` environment variable and then decodes the resulting string as JSON. That is, by passing the following line to the module configuration through the ``OUTPUT_FRAME`` environment variable:

.. code-block:: text

    '{"codec": "h264", "encoder_params": {"bitrate": 4000000}}'

The resulting configuration is presented in the following snippet:

.. code-block:: yaml

    parameters:
      output_frame:
        codec: h264
        encoder_params:
          bitrate: 4000000

