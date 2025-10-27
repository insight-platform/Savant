NvInfer Unit Overview and Common Configuration
==============================================

In Savant, DeepStream NvInfer is used to run inference on models prepared for Nvidia DeepStream. 
The framework supports four types of inference units:

- :doc:`30_dm` - detector unit (typically used for models producing bounding boxes, classes and confidence scores);
- :doc:`40_cm` - classifier unit (typically used for models producing classes and confidence scores);
- :doc:`43_am` - attribute model unit (typically used for models producing attributes like gender, age, etc.);
- :doc:`53_complexm` - complex model unit (typically used for models producing bounding boxes, classes and confidence scores, and attributes like keypoints, etc.).

Each of these units is configured a bit differently. In this section, we describe common parameters for all units.

About DeepStream NvInfer
------------------------

Nvinfer is a GStreamer plugin for running inference on models prepared for Nvidia DeepStream. In DeepStream it is configured with a configuration file. In Savant, we provide two ways for configuring the NvInfer unit:

- YAML-based configuration;
- a configuration file (you should not use it if we provide the required configuration parameter in YAML).

In the end, both of them lead to the same result: a generated configuration file in the model cache directory.

All of the NvInfer variants require the model to be specified. Savant supports local model files and remote model files. The first are convenient if/when they are burnt into the docker image. The second are convenient if/when they are downloaded from a remote location.

Regardless of the model source, NvInfer generates a TensorRT engine file for the model for the current hardware. If the engine file already exists, it will be used instead of generating a new one if it is compatible with the current configuration (hardware,batch size, precision, etc.).

Read more about working with models in :doc:`27_working_with_models`.

Examples of NvInfer unit configuration
--------------------------------------

Detector Unit
~~~~~~~~~~~~~

.. code-block:: yaml

    - element: nvinfer@detector
      name: yolov8n
      model:
        remote:
          url: s3://savant-data/models/yolov8n/yolov8n_000bcd6.zip
          checksum_url: s3://savant-data/models/yolov8n/yolov8n_000bcd6.md5
          parameters:
            endpoint: https://eu-central-1.linodeobjects.com
        format: onnx
        model_file: yolov8n.onnx
        batch_size: ${parameters.batch_size}
        input:
          shape: [3, 640, 640]
          maintain_aspect_ratio: true
          scale_factor: 0.0039215697906911373
        output:
          layer_names: [boxes, scores, classes]
          converter:
            module: savant.converter.yolo
            class_name: TensorToBBoxConverter
            kwargs:
              confidence_threshold: 0.2
              top_k: 1000
          objects:
            - class_id: ${parameters.detected_object.id}
              label: ${parameters.detected_object.label}
              selector:
                kwargs:
                  confidence_threshold: 0.6
                  nms_iou_threshold: 0.5
                  min_width: 30
                  min_height: 40


Classifier Unit
~~~~~~~~~~~~~~~

.. code-block:: yaml

    - element: nvinfer@classifier
      name: classifier_model
      model:
        format: onnx
        remote:
          url: s3://savant-data/models/classifier_model/classifier_model.zip
          checksum_url: s3://savant-data/models/classifier_model/classifier_model.md5
          parameters:
            endpoint: https://eu-central-1.linodeobjects.com
        model_file: classifier_model.onnx
        batch_size: ${parameters.batch_size}
        input:
          object: detector_model.Car
        output:
          layer_names: [ predictions/Softmax ]
          attributes:
            - name: class_label
              threshold: ${parameters.class_threshold}


Attribute Model Unit
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    - element: nvinfer@attribute_model
      name: age_gender
      model:
        remote:
          url: s3://savant-data/models/age_gender/age_gender.zip
          checksum_url: s3://savant-data/models/age_gender/age_gender.md5
          parameters:
            endpoint: https://eu-central-1.linodeobjects.com
        format: onnx
        config_file: age_gender_mobilenet_v2_dynBatch_config.txt
        batch_size: 16
        input:
          object: ${parameters.detection_model_name}.face
          preprocess_object_image:
            module: savant.input_preproc.align_face
            class_name: AlignFacePreprocessingObjectImageGPU
        output:
          layer_names: [ 'age', 'gender' ]
          converter:
            module:  samples.age_gender_recognition.age_gender_converter
            class_name: AgeGenderConverter
          attributes:
            - name: age
            - name: gender


Complex Model Unit
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    - element: nvinfer@complex_model
      name: yolov8nface
      model:
        remote:
          url: s3://savant-data/models/yolov8face/yolov8nface.zip
          checksum_url: s3://savant-data/models/yolov8face/yolov8nface.md5
          parameters:
            endpoint: https://eu-central-1.linodeobjects.com
        format: onnx
        config_file: yolov8n-face.txt
        batch_size: 16
        input:
          shape:
            - 3
            - ${parameters.detector_h}
            - ${parameters.detector_w}
        output:
          layer_names: [ 'output0' ]
          converter:
            module: savant.converter.yolo_v8face
            class_name: YoloV8faceConverter
            kwargs:
              confidence_threshold: 0.6
              nms_iou_threshold: 0.5
          objects:
            - class_id: 0
              label: face
              selector:
                module: savant.selector.detector
                class_name: MinMaxSizeBBoxSelector
                kwargs:
                  min_width: 40
                  min_height: 40
          attributes:
            - name: landmarks
