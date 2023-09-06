Element Groups
==============

A single module config can contain multiple variations of the pipeline through the use of conditionally initialized :py:class:`~savant.config.schema.ElementGroup`.

For example, the user may introduce alternative detector units without the need to duplicate all the other module parameters in a separate config file, as is done in the `traffic meter <https://github.com/insight-platform/Savant/tree/develop/samples/traffic_meter>`__ sample.

Here's a section of the module config from the beginning of the pipeline definition that adds a pyfunc unit as usual, and then defines a conditionally enabled group for the PeopleNet detector:

.. literalinclude:: ../../../samples/traffic_meter/module.yml
  :language: YAML
  :lines: 27-45

Later in the module config another group is defined for the YOLOV8m detector:

.. literalinclude:: ../../../samples/traffic_meter/module.yml
  :language: YAML
  :lines: 67-75

The groups are initialized based on whether their :py:attr:`~savant.config.schema.ElementGroup.init_condition` evaluates to `True` at the time of module initialization. In the example above which detector unit is initialized depends on the value of the ``DETECTOR`` environment variable: if it is set to ``peoplenet`` then the peoplenet detector is initialized, and similarly for the ``yolov8m`` and ``yolov8s`` detectors.

.. note::

    The behaviour is static, not dynamic. Once the pipeline is initialized its contents do not change, so if the value of ``DETECTOR`` is changed after the pipeline is initialized, the detectors will not be swapped out.

In this example the detector units are conditionally initialized based on the value of an environment variable, but any expression on the config variables can be used as the condition. Expression evaluation is done in the same way as for any other config variable, standard OmegaConf interpolation and resolvers can be used, along with :py:mod:`custom resolvers <savant.config>` from Savant.

In this manner any of the supported :doc:`/savant_101/12_pipeline` units can be conditionally initialized. In addition, the conditions can be applied to multiple units at the same time, by specifying them in the :py:attr:`~savant.config.schema.ElementGroup.elements` list of a single group.
