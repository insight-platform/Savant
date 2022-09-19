Examples
========

These configuration files for sample :repo-link:`Savant` modules can help to quickly setup some basic pipeline
or clear up a part of Savant config file format.

The same configs along with detailed READMEs can be found in the main repo.

Peoplenet detector
------------------

This module runs inference for a single detector model (Nvidia PeopleNet).

- ``remote`` node makes model to be automatically downloaded
- ``objects`` node omits `bags`` objects, which makes Savant filter them out

.. literalinclude:: ../../../samples/peoplenet_detector/module.yml
  :language: YAML

Deepstream test2
----------------

This module reproduces ``deepstream test 2`` app, which runs one primary 4-class detector
and 3 secondary classifiers for one of detected classes.

Use Deepstream configs for Caffe models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All Deepstream models can be quickly and easily included in a :repo-link:`Savant` pipeline by specifying
Deepstream nvinfer configs directly in the ``model.config_file`` config node.

Look in the main repo for the nvinfer config files used in this sample.

.. literalinclude:: ../../../samples/deepstream_test2/module.yml
  :language: YAML

Use generated TensorRT engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once user's models were converted into TensorRT engines, it is possible to simply use these engine files in the pipeline.

.. literalinclude:: ../../../samples/deepstream_test2/module-engines-config.yml
  :language: YAML

Configure Caffe models directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In case no nvinfer configs have already been written it is possible to set all the necessary values
straight in the Savant module config.

.. literalinclude:: ../../../samples/deepstream_test2/module-full-savant-config.yml
  :language: YAML

Use ETLT models
^^^^^^^^^^^^^^^

This same sample pipeline can be easily modified to use a different set of models, for example,
ETLT format models from Nvidia TAO Toolkit.

.. literalinclude:: ../../../samples/deepstream_test2/module-etlt-config.yml
  :language: YAML