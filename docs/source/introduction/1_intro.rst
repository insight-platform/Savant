Introduction
============

Savant is different from other deep learning, computer vision solutions like OpenCV, PyTorch, Nvidia DeepStream, etc. It holds a unique position among them because Savant is a framework, it defines the architecture, not only libraries. You fill Savant pipelines with your custom processing blocks, without the need to define or invent architecture, behaviors and interfaces (external and internal API).

Its difference may look complex from the first sight, but keep in mind that Savant is developed to provide means for building very efficient computer vision and deep learning pipelines.

Examples
--------

We developed example pipelines to help people get acquainted with Savant. Every example contains a short readme instruction on how to run it. Some examples are also accompanied by the articles published on Medium.

Examples are available on `GitHub <https://github.com/insight-platform/Savant/tree/develop/samples>`_.

Full-Featured Examples
^^^^^^^^^^^^^^^^^^^^^^

- People detector/tracker `example <https://github.com/insight-platform/Savant/tree/develop/samples/peoplenet_detector>`__, also covers OpenCV CUDA;
- DeepStream car detection, tracking and classification `example <https://github.com/insight-platform/Savant/tree/develop/samples/nvidia_car_classification>`__, which runs a primary detector, 3 classifiers and the Nvidia tracker;
- Ready-to-use traffic (people, cars, etc.) `counting <https://github.com/insight-platform/Savant/tree/develop/samples/traffic_meter>`__ with YOLOv8m/s or PeopleNet, and the Nvidia tracker;
- Age/gender prediction `example <https://github.com/insight-platform/Savant/tree/develop/samples/age_gender_recognition>`__ featuring facial detection (YOLOV5-Face), tracking, age/gender prediction (custom MobileNet V2) and predictions normalization based on tracking.
- Facial ReID `example <https://github.com/insight-platform/Savant/tree/develop/samples/face_reid>`__ demonstrating facial detection (YOLOV5-Face), ReID (AdaFace), and HNSWLIB lookups;
- Instance segmentation `example <https://github.com/insight-platform/Savant/tree/develop/samples/yolov8_seg>`__ with YOLOV8m-Seg.

Utility Examples
^^^^^^^^^^^^^^^^

- OpenCV CUDA MOG2 background subtraction `example <https://github.com/insight-platform/Savant/tree/develop/samples/opencv_cuda_bg_remover_mog2>`__;
- Conditional video processing `example <https://github.com/insight-platform/Savant/tree/develop/samples/conditional_video_processing>`__ using PeopleNet detector to turn processing on/off;
- Multiple RTSP streams processing `example <https://github.com/insight-platform/Savant/tree/develop/samples/multiple_rtsp>`__;
- RTSP Cam Compatibility Test `example <https://github.com/insight-platform/Savant/tree/develop/samples/rtsp_cam_compatibility_test>`__;
- GigE Vision Cam `example <https://github.com/insight-platform/Savant/tree/develop/samples/multiple_gige>`__.

Tutorials
----------

We created several tutorials to help you get started with Savant. These tutorials are designed for both x86+dGPU, and Jetson NX/AGX+ edge devices.

The tutorials are published on Medium:

- `Meet Savant: a New High-Performance Python Video Analytics Framework For Nvidia Hardware <https://blog.savant-ai.io/meet-savant-a-new-high-performance-python-video-analytics-framework-for-nvidia-hardware-22cc830ead4d?source=friends_link&sk=c7169b378b31451ab8de3d882c22a774>`_;
- `Building a 500+ FPS Accelerated Video Background Removal Pipeline with Savant and OpenCV CUDA MOG2 <https://blog.savant-ai.io/building-a-500-fps-accelerated-video-background-removal-pipeline-with-savant-and-opencv-cuda-mog2-441294570ac4?source=friends_link&sk=8cee4e671e77cb2b4bb36518619f9044>`_;
- `Building a High-Performance Car Classification Pipeline With Savant <https://blog.savant-ai.io/building-a-high-performance-car-classifier-pipeline-with-savant-b232461ad96?source=friends_link&sk=63cb289315679af83032ef5247861a2d>`_;
- `Efficient City Traffic Metering With PeopleNet/YOLOv8, Savant, And Grafana At Scale <https://blog.savant-ai.io/efficient-city-traffic-metering-with-peoplenet-yolov8-savant-and-grafana-at-scale-d6f162afe883?source=friends_link&sk=ab96c5ef3c173902559f213849dede9b>`_.

Savant simplifies everyday tasks, including data ingestion, stream processing, data output, and other operations encountered when building video analytics pipelines. With GStreamer technicalities being hidden by Savant, all the user writes is code and configuration specific to their video analytics application.

How The Pipeline Looks Like
---------------------------

In computer vision field the `pipeline` is a name commonly used to describe a set of steps computing various algorithms on images resulting in transformed images and their metadata. Such algorithms may include traditional computer vision algorithms like background subtraction and modern approaches based on DNN.

Savant is a framework to build pipelines. A pipeline implemented in Savant is called the `module`. We usually tell a `module` meaning `a running computer vision pipeline implemented in Savant`. Please remenber the term `module` as we use it in the text very often.

Every Savant module consists of two parts: YAML configuration and Python code. YAML is used to define module `units` (pipeline steps), and the code is used to implement custom metadata processing. Remember the term `unit` as we use it a lot in the text.

You don't know a lot about Savant, but we want you to take a look a pieces of YAML to get the idea how a units look like; later in the documentation, we will explore them in detail. There are essential omitted parts in every unit provided in the following code snippets; this is done with the intention of not overwhelming you with details.

We begin with a pipeline containing one unit - a detector:

.. code-block:: yaml

  pipeline:
    elements:
      - element: nvinfer@detector
        name: primary_detector
        output:
          objects:
            - class_id: 0
              label: car

Next, add a classifier model to the pipeline, which is used to classify cars into different colors:

.. code-block:: yaml

  elements:
    - element: nvinfer@detector
      name: primary_detector
      output:
        objects:
          - class_id: 0
            label: car
    - element: nvinfer@attribute_model
      name: secondary_carcolor
      input:
        objects: primary_detector.car
      output:
        attributes:
          - name: car_color

Next, add custom processing in Python, e.g., count the number of detected cars of each color:

.. code-block:: yaml

  elements:
    - element: nvinfer@detector
      name: primary_detector
      output:
        objects:
          - class_id: 0
            label: car
    - element: nvinfer@attribute_model
      name: secondary_carcolor
      input:
        objects: primary_detector.car
      output:
        attributes:
          - name: car_color
    - element: pyfunc
      name: car_counter
      module: module.car_counter
      class_name: CarCounter

The contents of ``module/car_counter.py`` would be:

.. code-block:: python

    from collections import Counter
    from savant.deepstream.meta.frame import NvDsFrameMeta
    from savant.deepstream.pyfunc import NvDsPyFuncPlugin

    counter = Counter()

    CAR_COLOR_ELEMENT_NAME = 'secondary_carcolor'
    CAR_COLOR_ATTR_NAME = 'car_color'

    class CarCounter(NvDsPyFuncPlugin):
        def process_frame(self, buffer: Gst.Buffer, frame_meta: NvDsFrameMeta):
            for obj_meta in frame_meta.objects:
                car_color_attr = obj_meta.get_attr_meta(CAR_COLOR_ELEMENT_NAME, CAR_COLOR_ATTR_NAME)
                counter[car_color_attr.value] += 1

This is what a typical Savant pipeline looks like. Other examples of pipelines can be found in the `examples <https://github.com/insight-platform/Savant/tree/develop/samples>`_ directory.

Next, we are moving to the part describing how to launch `modules`.

Performance
-----------

Wait, you may be confused about the performance and Python in the same context. Python is usually not corresponding to performance. Let's discuss it in the context of Savant. Basically Savant is built on:

- GStreamer (C/C++);
- Nvidia DeepStream (C/C++);
- Rust Core Library (Savant-RS) with Python bindings;
- Custom C/C++ GStreamer plugins;
- OpenCV CUDA (C/C++/Python bindings);
- Python NumPy;
- Python Numba (native, nogil);
- Python box moving code.

We focus on developing critical parts of Savant with technologies capable of unlocking GIL and enabling parallelization. We consider Python to be an excellent box-moving flow control solution. The custom code developers create can be developed in pure or efficient Python with FFI code, unlocking parallel execution.

Savant makes it possible for every pipeline stage to run in parallel: it is up to the developer to deliver efficient code, unlocking parallelization.
