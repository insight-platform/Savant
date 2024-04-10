Working With Models
===================

In order to understand how to properly prepare a model for use in a pipeline, let's first go over the main configuration elements responsible for the model inference. Some parameters have been omitted for simplicity, but you will learn about specific and necessary parameters for specific types of models in dedicated sections.

The listing below represents a typical Savant inference node:

.. code-block:: yaml

    - element: nvinfer@detector
      name: detection_model
      model:
        format: etlt
        remote:
          url: "https://127.0.0.1/models/detection_model.zip"
        local_path: /opt/aispp/models/detection_model
        model_file: resnet18_dashcamnet_pruned.etlt
        engine_file: resnet18_dashcamnet_pruned.etlt_b1_gpu0_int8.engine
        batch_size: 1
        precision: int8
        int8_calib_file: dashcamnet_int8.txt
        input:
          layer_name: input_1
          shape: [3, 544, 960]
          scale_factor: 0.00392156862745098
        output:
          layer_names: [output_cov/Sigmoid, output_bbox/BiasAdd]

The ``element`` section specifies the type of a pipeline unit. There are 4 types of units for defining models: :doc:`detector </savant_101/30_dm>`, :doc:`classifier </savant_101/40_cm>`, :doc:`attribute_model </savant_101/43_am>`, instance_segmentation, and :doc:`complex_model </savant_101/53_complexm>`.

The ``name`` parameter defines the name of the unit. The ``name`` is used by the downstream pipeline units to refer to the objects that the unit produces. This parameter is also used to construct the path to the model files, see the ``local_path`` parameter.

The ``format`` parameter specifies the format in which the model is provided. The parameter is used to build the TensorRT engine and can be omitted if a pre-built engine file is provided. The supported formats and the peculiarities of specifying certain parameters depending on the model format are described below.

The ``model_file`` parameter defines the name of the file with the model. The name is specified as a base name, not a full path.

The ``engine_file`` parameter defines the name for the TensorRT-generated engine file. If this parameter is set, then when the pipeline is launched, the presence of this file is checked first, and if it is present, the model will be loaded from it.

If the prepared model engine file does not exist, then the pipeline will generate the engine for the model. If you are not using a specially generated, pre-created TensorRT engine file, it is recommended not to set this field: the name will be generated automatically.

The ``remote`` section specifies a URL and credentials for accessing a remote model storage. Full description below. Savant supports downloading the models from remote locations so you can easily update them without rebuilding docker images.

The ``local_path`` parameter specifies the path to the model files. It can be omitted, then the path will be automatically generated according to the following rule ``<model_path>/<name>``, where ``<model_path>`` is a global parameter specifying the location of all model files, set in the :ref:`parameters <savant_101/12_module_definition:parameters>` section, and ``<name>`` is the name of the element.

The ``batch_size`` parameter defines a batch size dimension for processing frames by the model (by default 1).

The ``precision`` parameter defines the data format to be used by inference of the model. Possible values are ``fp32``, ``fp16``, ``int8``. The parameter is important for TensorRT engine creation and affects the speed of inference. ``int8`` inference is faster than ``fp16``, but requires a calibration file. ``fp16`` is faster than ``fp32``, TensorRT can perform the conversion automatically, with little or no degradation in model accuracy, so ``fp16`` is set by default.

The ``int8_calib_file`` defines the name of the calibration file in case the model ``precision`` is set to ``int8``.

The ``input`` section describes the model input: names of input layers, dimensionality, etc. The mandatory or optional nature of the parameters in this section depends on the model format, as well as on the type of model. This section will be covered in more detail later, when describing model formats or types of models.

The ``output`` section describes the model output: names of output layers, converters, selectors, etc. The mandatory or optional nature of the parameters in this section depends on the model format, as well as on the type of model. This section will be covered in more detail later, when describing model formats.

To accelerate inference in the framework, NVIDIA TensorRT is used. To use a model in a pipeline, it must be presented in one of the formats supported by TensorRT:

ONNX
----

ONNX is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers. This format is recommended as the to-go format for models.

To export a model from one of the most popular frameworks, you can refer to the instructions or examples provided below:

* `PyTorch <https://pytorch.org/docs/stable/onnx.html>`_;
* `TensorFlow <https://github.com/onnx/tensorflow-onnx>`_;
* `MXNet <https://mxnet.apache.org/versions/1.9.1/api/python/docs/tutorials/deploy/export/onnx.html>`_.

Usage example:

.. code-block:: yaml

  - element: nvinfer@detector
    name: DetectionModel
    model:
      format: onnx
      model_file: detection_model.onnx

If the model has non-standard outputs (outputs that cannot be automatically converted by DeepStream into meta information), then it is also necessary to specify the name or names of the output layers in the output section.

.. code-block:: yaml

  - element: nvinfer@detector
    name: DetectionModel
    model:
      format: onnx
      model_file: detection_model.onnx
      output:
        layer_names: [output]

UFF
---

UFF is an intermediate format for representing a model between TensorFlow and TensorRT. Users who use the TensorFlow framework can convert their models to the UFF format using the UFF converter. If you are using a model in the UFF format, you must specify the name of the input layer (``layer_name``) and the input dimensionality of the model (``shape``) in the ``input`` section, as well as the name(s) of the resulting layer(s) (``layer_names``) in the ``output`` section.

.. code-block:: yaml

  - element: nvinfer@detector
    name: DetectionModel
    model:
      format: uff
      model_file: detection_model.uff
      input:
        layer_name: input_1
        shape: [3, 544, 960]
      output:
        layer_names: [output_cov/Sigmoid, output_bbox/BiasAdd]

This format will no longer be supported by future releases of TensorRT (`Tensor RT release notes <https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html#tensorrt-9>`_).

Caffe
-----

If you have a model trained using the Caffe framework, then you can save your model in the ``caffemodel`` format.

.. code-block:: yaml

  - element: nvinfer@detector
    name: DetectionModel
    model:
      format: caffe
      model_file: detection_model.caffemodel
      proto_file: resnet.prototxt
      output:
        layer_names: [output_cov/Sigmoid, output_bbox/BiasAdd]


This format will no longer be supported by future releases of TensorRT (`Tensor RT release notes <https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html#tensorrt-9>`_).

NVIDIA TAO Toolkit
------------------

The NVIDIA TAO Toolkit is a set of training tools that requires minimal effort to create computer vision neural models based on user's own data. Using the TAO toolkit, users can perform transfer learning from pre-trained NVIDIA models to create their own model.

After training the model, you can download it in the ``etlt`` format and use this file for model inference in the Savant framework. If you are using a model in the ``etlt`` format, you must specify the name of the input layer (``layer_name``) and the input dimensionality of the model (``shape``) in the ``input`` section, as well as the name(s) of the resulting layer(s) (``layer_names``) in the output section.

.. code-block:: yaml

  - element: nvinfer@detector
    name: DetectionModel
    model:
      format: etlt
      model_file: detection_model.etlt
      input:
        layer_name: input_1
        shape: [3, 544, 960]
      output:
        layer_names: [output_cov/Sigmoid, output_bbox/BiasAdd]

Custom CUDA Engine
------------------

For all the above-mentioned variants of specifying the model, during the first launch, an engine file will be generated using TensorRT with automatic parsing of the model. When the model is very complex or requires some custom plugins or layers, you can generate the engine file yourself using the TensorRT API and specifying the library and the name of the function that generates the engine (`Using custom model with DeepStream <https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_using_custom_model.html>`_).

.. code-block:: yaml

  - element: nvinfer@detector
    name: DetectionModel
    model:
      format: custom
      custom_config_file: yolov2-tiny.cfg
      custom_lib_path: libnvdsinfer_custom_impl_Yolo.so
      engine_create_func_name: NvDsInferYoloCudaEngineGet

Build Model Engine
------------------

Savant uses the DeepStream element ``nvinfer`` to perform model inferencing. Under the hood, nvinfer uses TensorRT to facilitate high-performance machine learning inference. Any of the supported model types (ONNX, UFF, TAO) must be converted to the TensorRT engine for use in the pipeline.

The TensorRT engine, unlike the model file (ONNX, UFF, TAO), is not a universal model representation, but a device-specific optimized representation. That is, it cannot be transferred between different devices. This justifies the practice of generating the TensorRT engine when initializing the nvinfer element. When the pipeline with a model is started, if the engine is missing, it will be generated based on the provided config (with a given batch size, etc.) from the model source file (ONNX, UFF, TAO). This process can take more than 10 minutes for complex models like YOLO. Savant makes it easy to cache model files, including those generated by the TensorRT. If the engine is submitted and matches the configuration, the model engine generation step will be skipped and pipelines will start immediately.

Savant supports explicit engine generation as a separate, preliminary step of running the Savant module pipeline. The generation is done by running a simplified pipline that contains a model element (nvinfer). You can use the :py:func:`savant.deepstream.nvinfer.build_engine.build_engine` function in your code for this purpose, or you can run the generation step of all the module engines via the main module entry point specifying the option ``--build-engines``

.. code-block:: bash

  python -m savant.entrypoint --build-engines path/to/module/config.yml

For example, you can build the model engines used in the `Nvidia car classification <https://github.com/insight-platform/Savant/tree/develop/samples/nvidia_car_classification>`_ example with the following command (you are expected to be in Savant/ directory):

.. code-block:: bash

    ./scripts/run_module.py --build-engines samples/nvidia_car_classification/module.yml

You can also use the ``trtexec`` tool from the TensorRT package to generate an engine file. But you need to understand exactly what parameters you need to use to generate a suitable for ``nvinfer`` engine file. And you can't use ``trtexec`` if you need a custom engine generator.

Example of using ``trtexec`` to build engine for ONNX model:

.. code-block:: bash

    /usr/src/tensorrt/bin/trtexec --onnx=/cache/models/custom_module/model_name/model_name.onnx --saveEngine=/cache/models/custom_module/model_name/model_name.onnx_b16_gpu0_fp16.engine --minShapes='images':1x3x224x224 --optShapes='images':16x3x224x224 --maxShapes='images':16x3x224x224 --fp16 --workspace=6144 --verbose

Using Pre-built Model Engine
----------------------------

If you have a pre-built engine file, you can use it in the pipeline without having to include the original model file and set some of the parameters required to generate the engine from the model file (e.g. input and output layer names for UFF model). The engine file must be placed in the model file directory. The name of the engine file must be specified in the ``engine_file`` parameter of the model configuration.

.. code-block:: yaml

  - element: nvinfer@detector
    name: detection_model
    model:
      engine_file: detection_model.onnx_b1_gpu0_fp16.engine

We recommend using the ``nvinfer`` name format for the engine file: ``{model_name}_b{batch_size}_gpu{gpu_id}_{precision}.engine``. This allows you to easily understand the configuration of the model engine and saves you from having to set ``batch-size`` and ``precision`` separately in the model config.

Working With Remote Models
--------------------------

Currently, there are three data transfer protocols supported: S3, HTTP(S), and FTP. By specifying the URL of the archive file, you can use models that are stored remotely.

.. code-block:: yaml

  - element: nvinfer@detector
    name: Primary_Detector
    model:
      format: caffe
      remote:
        url: s3://savant-data/models/Primary_Detector/Primary_Detector.zip
        checksum_url: s3://savant-data/models/Primary_Detector/Primary_Detector.md5
        parameters:
          endpoint: https://eu-central-1.linodeobjects.com

In this example, in the remote section, we specify:

* ``url`` - specifies where to download the archive file from;
* ``checksum_url`` - specifies the file that stores the md5 checksum for the archive; if the archive has not been updated, it will not be downloaded during the next module launch;
* ``parameters`` - a section that allows you to specify additional parameters for the S3, HTTP(S), or FTP protocols:
   * S3 protocol parameters: ``access_key``, ``secret_key``, ``endpoint``, ``region``;
   * HTTP(S) protocol parameters: ``username``, ``password``;
   * FTP protocol parameters: ``username``, ``password``.

All necessary files (model file in one of the formats described above, configuration, calibration, and other files that you specify when configuring the model) must be archived using one of the archivers (``gzip``, ``bzip2``, ``xz``, ``zip``). The archive must contain all necessary model files.

You can download an example model archive used in the `Nvidia car classification <https://github.com/insight-platform/Savant/tree/develop/samples/nvidia_car_classification>`_ example with the following command:

.. code-block:: bash

  aws --endpoint-url=https://eu-central-1.linodeobjects.com s3 cp s3://savant-data/models/Primary_Detector/Primary_Detector.zip .

You can find an example of using this model archive at the following `link <https://github.com/insight-platform/Savant/blob/develop/samples/nvidia_car_classification/module.yml#L46>`_.
