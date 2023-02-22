Module overview
===============

A module is an executable unit that can be deployed and executed on Nvidia edge devices
or in the cloud on x86 servers with discrete GPUs. Module contents are declared in a YAML
configuration file.

Module configuration
--------------------

In general, detailed description of the configuration file format is presented
in the API reference :ref:`reference/api/module_config:Module configuration`.

On the top level, every module should have a ``name``, which is an arbitrary string, a ``pipeline`` (pipeline definition is discussed in details in :doc:`pipeline`) and will probably have some ``parameters``.

Module parameters overview
^^^^^^^^^^^^^^^^^^^^^^^^^^

Any number of :py:attr:`~savant.config.schema.Module.parameters` can be set in the ``parameters`` section of the module configuration file, including user-defined ones.

The following parameters are defined for a Savant module by default:

.. literalinclude:: ../../../savant/config/default.yml
  :language: YAML
  :lines: 1-75

.. note::

  Config values set in the form of ``${value}`` are using OmegaConf
  `variable interpolation <https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#variable-interpolation>`_.

Accessing parameter values at runtime is possible through an object of :py:class:`~savant.parameter_storage.ParameterStorage` class.
There's no need for a user to create this object, simply get the reference from :py:class:`~savant.parameter_storage.param_storage`

.. code-block:: python

  from savant.parameter_storage import param_storage

  parameter_value = param_storage()['parameter_name']

Dynamic parameters
^^^^^^^^^^^^^^^^^^

A special case for module parameters is :py:attr:`~savant.config.schema.Module.dynamic_parameters`. These parameters are declared in
a separate config section ``dynamic_parameters`` and will have their value watched and updated according to current value
in dynamic parameter storage.

.. note::

  Currently ``etcd`` is supported as dynamic parameter storage. Etcd connection requires
  parameter values under ``parameters.etcd_config`` to be valid.

Configuring frame output
^^^^^^^^^^^^^^^^^^^^^^^^

In case a Savant module is intended to run alongside **image-** or **video-files** sink :ref:`adapters <main_concepts/adapters:Adapters overview>` its ``output_frame`` parameter must be configured correctly in order to get frame data in the output messages.

1. ``parameters.output_frame.codec`` config node determines which codec will be used to encode output frames. Possible values are **h264**, **h265**, **jpeg**, **png** and **raw-rgba** to skip encoding. For example:

  .. code-block:: YAML

    parameters:
      output_frame:
        codec: h264

2. ``parameters.output_frame.encoder_params`` config node allows passing additional encoder-specific parameters to the encoder pipeline element. These are:

  - **nvv4l2h264enc** for **h264** codec
  - **nvv4l2h265enc** for **h265** codec
  - **nvjpegenc** or **jpegenc** for **jpeg** codec
  - **pngenc** for **png** codec

  Example:

  .. code-block:: YAML

    parameters:
      output_frame:
        codec: h264
        encoder_params:
          bitrate: 4000000
          profile: 4

  .. code-block:: YAML

    parameters:
      output_frame:
        codec: jpeg
        encoder_params:
          quality: 90

**gst-inspect-1.0** utility inside the module container provides full list of properties for each encoder element.

Module run environment
----------------------

Modules are run in docker containers. If the module does not require any additional dependencies
base Savant docker image can be used to run it. Otherwise custom run environment must be provided by
extending base Savant docker image. There are three base images:

#. For Nvidia DGPUs on x86 architecture
#. For Deepstream 6.1+ capable Nvidia edge devices (Jetson Xavier/Orin)
#. For Deepstream 6.0.1 capped Nvidia edge devices (Jetson Nano/TX)

since module execution is provided for three families of target devices
(`docker registry <https://github.com/orgs/insight-platform/packages?repo_name=savant>`_).

A module should include two main directories:

#. The directory with the configuration file and source files with custom python code.
   Custom C/C++ functions must be compiled for the target platform and the libraries must be placed
   into the ``lib`` subdirectory.

#. The directory with models. Models can be in any format,
   but if you want to include a model as a model element,
   the model must be in one of the formats supported by
   `Gst-nvinfer <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html>`_
   (Caffe, ONNX, UFF, TAO) and satisfy following requirements:

   #. The model must have layers supported by TensorRT 8.
   #. The model must have a single input layer.
   #. There may be more that one output layer with arbitrary names.
