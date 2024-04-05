.. Savant documentation master file,
   should at least contain the root `toctree` directive.

Welcome To Savant Documentation
===================================

:repo-link:`Savant` is an open-source, high-level Python/C++/Rust framework for building real-time, streaming, highly efficient computer vision AI applications on the Nvidia stack. It makes it possible to develop dynamic, fault-tolerant inference pipelines that utilize the best Nvidia approaches for data center and edge accelerators very quickly. Savant supports both discrete Nvidia GPUs and Jetson edge devices.

* **Repository**: https://github.com/insight-platform/Savant
* **License**: Apache 2.0

Why Savant Was Developed?
-------------------------

We developed it to give deep learning and computer vision engineers a pipeline development framework that is both easy-to-use and implements the best Nvidia technologies.

Savant is a high-level framework on Nvidia `DeepStream SDK <https://developer.nvidia.com/deepstream-sdk>`_, which hides complexity and provides practical functions for implementing blazingly fast streaming AI applications.

In Savant, the pipeline is a sequence of declarative (YAML) blocks with user-defined functions in Python.

Key Features
------------

Savant is packed with several killer features which skyrocket the development of Deepstream applications.

🔧 All You Need for Building Real-Life Applications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Savant supports everything you need for developing advanced pipelines: detection, classification, segmentation, tracking, and custom pre- and post-processing for meta and images.

We have implemented samples demonstrating pipelines you can build with Savant. Visit the [samples](samples) folder to learn more.

🚀 High Performance
^^^^^^^^^^^^^^^^^^^

Savant is designed to be fast: it works on top of DeepStream - the fastest SDK for video analytics. Even the heavyweight segmentation models can run in real-time on Savant. See the [Performance Regression Tracking Dashboard](docs/performance.md) for the latest performance results.

🌐 Works On Edge and Data Center Equipment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The framework supports running the pipelines on both Nvidia's edge devices (Jetson Family) and data center devices (Tesla, Quadro, etc.) with minor or zero changes.

❤️ Cloud-Ready
^^^^^^^^^^^^^^

Savant pipelines run in Docker containers. We provide images for x86+dGPU and Jetson hardware.

### ⚡ Low Latency and High Capacity Processing

Savant can be configured to execute a pipeline in real-time, skipping data when running out of capacity or in high capacity mode, which guarantees the processing of all the data, maximizing the utilization of the available resources.

🤝 Ready-To-Use API
^^^^^^^^^^^^^^^^^^^

A pipeline is a self-sufficient service communicating with the world via high-performance streaming API. Whether developers use provided adapters or Client SDK, both approaches use the API.

📁 Advanced Data Protocol
^^^^^^^^^^^^^^^^^^^^^^^^^

The framework universally uses a common protocol for both video and metadata delivery. The protocol is highly flexible, allowing video-related information alongside arbitrary structures useful for IoT and 3rd-party integrations.

⏱ OpenTelemetry Support
^^^^^^^^^^^^^^^^^^^^^^^^

In Savant, you can precisely instrument pipelines with OpenTelemetry: a unified monitoring solution. You can use sampled or complete traces to balance the performance and precision. The traces can span from edge to core to business logic through network and storage because their propagation is supported by the Savant protocol.

📊 Prometheus Support
^^^^^^^^^^^^^^^^^^^^^

Savant pipelines can be instrumented with Prometheus: a popular monitoring solution. Prometheus is a great choice for monitoring the pipeline's performance and resource utilization.

🧰 Client SDK
^^^^^^^^^^^^^

We provide Python-based SDK to interact with Savant pipelines (ingest and receive data). It enables simple integration with 3rd-party services. Client SDK is integrated with OpenTelemetry providing programmatic access to the pipeline traces and logs.

🧘 Development Server
^^^^^^^^^^^^^^^^^^^^^

Software development for vanilla DeepStream is a pain. Savant provides a Development Server tool, which enables dynamic reloading of changed code without pipeline restarts. It helps to develop and debug pipelines much faster. Altogether with Client SDK, it makes the development of DeepStream-enabled applications really simple. With the Development Server, you can develop remotely on a Jetson device or server right from your IDE.

🔀 Dynamic Sources Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In Savant, you can dynamically attach and detach sources and sinks to the pipeline without reloading. The framework resiliently handles situations related to source/sink outages.

🏹 Handy Source and Sink Adapters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The communication interface is not limited to Client SDK: we provide several ready-to-use adapters, which you can use "as is" or modify for your needs.

🎯 Dynamic Parameters Ingestion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Advanced ML pipelines may require information from the external environment for their work. The framework enables dynamic configuration of the pipeline with:

- ingested frame attributes passed in per-frame metadata;
- Etcd's attributes watched and instantly applied;
- 3rd-party attributes, which are received through user-defined functions.

🖼 OpenCV CUDA Support
^^^^^^^^^^^^^^^^^^^^^^

Savant supports custom OpenCV CUDA bindings enabling operations on DeepStream's in-GPU frames with a broad range of OpenCV CUDA functions: the feature helps in implementing highly efficient video transformations, including but not limited to blurring, cropping, clipping, applying banners and graphical elements over the frame, and others. The feature is available from Python.

🔦 PyTorch Support
^^^^^^^^^^^^^^^^^^

Savant supports PyTorch, one of the most popular ML frameworks. It enables the developer to use ready-to-use PyTorch models from PyTorchHub, a large number of code samples, and reliable extensions. The integration is highly efficient: it allows running inference on GPU-allocated images and processing the results in GPU RAM, avoiding data transfers between CPU and GPU RAM.

🔢 CuPy Support For Post-Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Savant supports CuPy: a NumPy-like library for GPU-accelerated computing. It enables the developer to implement custom post-processing functions in Python, executed in GPU RAM, avoiding data transfers between CPU and GPU RAM. The feature allows for accessing model output tensors directly from GPU RAM, which helps implement heavy-weight custom post-processing functions.

The integration also provides a conversion for in-GPU data between CuPy, OpenCV, and PyTorch in-GPU formats.

↻ Rotated Detection Models Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We frequently deal with the models resulting in bounding boxes rotated relative to a video frame (oriented bounding boxes). For example, it is often the case with bird-eye cameras observing the underlying area from a high point.

Such cases may require detecting the objects with minimal overlap. To achieve that, special models are used which generate bounding boxes that are not orthogonal to the frame axis. Take a look at `RAPiD <https://vip.bu.edu/projects/vsns/cossy/fisheye/rapid/>`_ to find more.

⇶ Parallelization
^^^^^^^^^^^^^^^^^

Savant supports processing parallelization; it helps to utilize the available resources to the maximum. The parallelization is achieved by running the pipeline stages in separate threads. Despite flow control-related Python code is not parallel; the developer can utilize GIL-releasing mechanisms to achieve the desired parallelization with NumPy, Numba, or custom native code in C++ or Rust.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Introduction

   introduction/1_intro
   introduction/2_running
   introduction/3_hardware_compatibility


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   getting_started/0_configure_prod_env
   getting_started/1_configure_dev_env
   getting_started/2_module_devguide


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Savant 101

   savant_101/00_streaming_model
   savant_101/10_adapters
   savant_101/12_module_definition
   savant_101/12_pipeline
   savant_101/12_video_processing
   savant_101/12_var_interpolation
   savant_101/12_metadata
   savant_101/25_top_level_roi
   savant_101/27_working_with_models
   savant_101/30_dm
   savant_101/40_cm
   savant_101/43_am
   savant_101/53_complexm
   savant_101/55_preprocessing.rst
   savant_101/60_nv_trackers
   savant_101/70_python
   savant_101/75_working_with_metadata
   savant_101/80_opencv_cuda
   savant_101/80_map
   savant_101/90_draw_func

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Advanced Topics

   advanced_topics/0_extra_image
   advanced_topics/0_batching
   advanced_topics/0_pipeline_stream_limit
   advanced_topics/0_pipeline_benchmarking
   advanced_topics/0_dead_stream_eviction
   advanced_topics/1_custom_tracking
   advanced_topics/2_element_group
   advanced_topics/3_custom_roi
   advanced_topics/3_frame_filtering
   advanced_topics/3_skipping_frames
   advanced_topics/3_hybrid_pipelines
   advanced_topics/4_etcd
   advanced_topics/6_chaining
   advanced_topics/8_ext_systems
   advanced_topics/9_dev_server
   advanced_topics/9_open_telemetry
   advanced_topics/9_input_json_metadata
   advanced_topics/10_client_sdk
   advanced_topics/11_memory_representation_function.rst
   advanced_topics/12_torch_hub.rst

.. toctree::
   :maxdepth: 0
   :hidden:
   :caption: Cookbook

   recipes/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   reference/api/index

.. toctree::
   :maxdepth: 0
   :hidden:
   :caption: Savant Development

   savantdev/0_configure_doc_env


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
