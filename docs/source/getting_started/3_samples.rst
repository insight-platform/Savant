Savant Samples Catalog
======================

This document provides a comprehensive overview of all available Savant samples. Each sample demonstrates specific computer vision and video processing capabilities using different models, adapters, and platform configurations.

.. note::
   Platform support is indicated as follows:
   
   - **X86 + L4T**: Both x86 and Jetson (L4T) platforms supported
   - **X86 only**: Only x86 platform supported
   - **L4T only**: Only Jetson (L4T) platform supported

   When not specified, the sample is implemented only for X86 platform by some reason.

Computer Vision and AI Samples
-------------------------------

Face Detection and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Age Gender Recognition
^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Face detection using YOLOv8-Face model with 5 face landmarks (eyes, nose, mouth)
- Age and gender estimation for detected faces
- Face orientation calculation using landmarks
- Face tracking with Nvidia Tracker

**Auxiliary Features**:

- Image preprocessing for model input
- Real-time and capacity processing modes
- Performance benchmarking support

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Age Gender Recognition <https://github.com/insight-platform/Savant/tree/develop/samples/age_gender_recognition>`_

Face ReID
^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Facial recognition and re-identification using YOLOv8-Face and AdaFace models
- Face gallery indexing and matching
- Doorbell security system demonstration

**Auxiliary Features**:

- Index builder for face gallery management
- Face preprocessing and feature vector extraction
- HNSWLIB-based face matching

**Adapters Used**:

- :ref:`Video files source adapter <video_file_source_adapter>` (for index building)
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>` (for demo)
- Client SDK integration

**Link**: `Face ReID <https://github.com/insight-platform/Savant/tree/develop/samples/face_reid>`_

People Detection and Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PeopleNet Detector
^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Person and face detection using Nvidia PeopleNet model
- GPU-accelerated face blurring with OpenCV CUDA
- Body-face matching and tracking

**Auxiliary Features**:

- Real-time and capacity processing modes
- Flickering reduction with simple tracker
- Performance benchmarking

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `PeopleNet Detector <https://github.com/insight-platform/Savant/tree/develop/samples/peoplenet_detector>`_

Traffic and Line Crossing Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Traffic Meter
^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Pedestrian line crossing detection and counting
- Multiple detector model support (PeopleNet, YOLOv8m, YOLOv8s)
- Direction-aware crossing detection
- Grafana dashboard integration

**Auxiliary Features**:

- DeepStream-Yolo integration
- Graphite metrics storage
- Real-time dashboard visualization

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Traffic Meter <https://github.com/insight-platform/Savant/tree/develop/samples/traffic_meter>`_

Intersection Traffic Meter
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Vehicle crossing detection at city intersections (cars, trucks, buses)
- Polygon-based intersection area definition
- YOLOv8 model for vehicle detection
- Multi-source and multi-polygon counting

**Auxiliary Features**:

- DeepStream-Yolo integration
- Grafana dashboard with metrics visualization
- Graphite storage backend

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Intersection Traffic Meter <https://github.com/insight-platform/Savant/tree/develop/samples/intersection_traffic_meter>`_

Fisheye Line Crossing
^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Line crossing detection for fisheye camera footage
- YOLOv7 OBB (Oriented Bounding Box) detector
- Similari tracking library integration
- Rotated bounding box detection

**Auxiliary Features**:

- Grafana dashboard integration
- Multi-source line crossing analytics
- Performance benchmarking

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Fisheye Line Crossing <https://github.com/insight-platform/Savant/tree/develop/samples/fisheye_line_crossing>`_

Area Object Counting
^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- People counting within user-defined areas
- Multi-area simultaneous monitoring
- Real-time area occupancy display

**Auxiliary Features**:

- Configurable area definitions
- Real-time visualization
- Multi-source processing

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Area Object Counting <https://github.com/insight-platform/Savant/tree/develop/samples/area_object_counting>`_

Object Detection and Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

License Plate Recognition
^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Car detection using YOLOv8 models
- License plate detection using Nvidia LPD model
- License plate text recognition using Nvidia LPR model
- Vehicle and plate tracking

**Auxiliary Features**:

- Multi-stage detection pipeline
- US license plate dictionary support
- Performance benchmarking

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `License Plate Recognition <https://github.com/insight-platform/Savant/tree/develop/samples/license_plate_recognition>`_

Nvidia Car Classification
^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Car detection and tracking
- Multi-attribute classification (type, color, make)
- Reproduces deepstream-test2 functionality

**Auxiliary Features**:

- Multiple classification models
- Track ID visualization
- Performance benchmarking

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Nvidia Car Classification <https://github.com/insight-platform/Savant/tree/develop/samples/nvidia_car_classification>`_

RT-DETR R50 Demo
^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Object detection using RT-DETR model
- Real-time detection transformer architecture
- DeepStream-Yolo integration

**Auxiliary Features**:

- ONNX model format support
- Performance optimization
- Real-time processing capability

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `RT-DETR <https://github.com/insight-platform/Savant/tree/develop/samples/rtdetr>`_

Advanced Computer Vision
~~~~~~~~~~~~~~~~~~~~~~~~

YOLOv8 Instance Segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Person instance segmentation using YOLOv8-seg model
- GPU and CPU converter options
- Complex model output processing

**Auxiliary Features**:

- CuPy GPU acceleration support
- NumPy/Numba CPU processing
- cv2.cuda.GpuMat rendering
- Performance optimization options

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `YOLOv8 Segmentation <https://github.com/insight-platform/Savant/tree/develop/samples/yolov8_seg>`_

Keypoint Detection
^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Human body keypoint detection using YOLOv8n-pose model
- 17-point body pose estimation
- Real-time pose visualization

**Auxiliary Features**:

- Ultralytics model integration
- ONNX export pipeline
- Performance benchmarking

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Keypoint Detection <https://github.com/insight-platform/Savant/tree/develop/samples/keypoint_detection>`_

NanoSAM
^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Object segmentation using NanoSAM model
- Point-based object identification
- Custom model input handling
- Multi-object color-coded segmentation

**Auxiliary Features**:

- Four-point interactive segmentation
- Gradient mask visualization
- Custom pyfunc integration
- TensorRT engine customization

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `NanoSAM <https://github.com/insight-platform/Savant/tree/develop/samples/nanosam>`_

Image Processing and Enhancement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AnimeGAN
^^^^^^^^

**Platform Support**: X86 only

**Main Features**:

- Anime-style image transformation using AnimeGANv2
- Hayao Miyazaki anime style application
- Video style transfer

**Auxiliary Features**:

- PyTorch to ONNX conversion pipeline
- ONNX model simplification
- Frame replacement demonstration

**Adapters Used**:

- :ref:`Multi-stream source adapter <multi_stream_source_adapter>`
- :ref:`Video files sink adapter <video_file_sink_adapter>`

**Link**: `AnimeGAN <https://github.com/insight-platform/Savant/tree/develop/samples/animegan>`_

Super Resolution
^^^^^^^^^^^^^^^^

**Platform Support**: X86 only

**Main Features**:

- Video super-resolution using NinaSR models
- 360p to 1080p upscaling demonstration
- Multiple scale factor support (2x, 3x, 4x)

**Auxiliary Features**:

- TorchSR model integration
- Lightweight neural network approach
- Quality enhancement visualization

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Super Resolution <https://github.com/insight-platform/Savant/tree/develop/samples/super_resolution>`_

OpenCV CUDA Background Removal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Real-time background removal using OpenCV CUDA MOG2
- High-performance processing (500+ FPS capability)
- GPU-accelerated background segmentation

**Auxiliary Features**:

- Hardware acceleration optimization
- Real-time and capacity processing modes
- Performance benchmarking

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `OpenCV CUDA Background Removal <https://github.com/insight-platform/Savant/tree/develop/samples/opencv_cuda_bg_remover_mog2>`_

Panoptic Driving Perception
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Driving scene understanding using YOLOP model
- Object detection and semantic segmentation
- PyTorch inference in Savant
- GPU memory interaction demonstration

**Auxiliary Features**:

- Torch hub integration
- Multi-task learning approach
- Real-time driving analysis

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Panoptic Driving Perception <https://github.com/insight-platform/Savant/tree/develop/samples/panoptic_driving_perception>`_

Streaming and Adapter Samples
------------------------------

Source Adapters
~~~~~~~~~~~~~~~

MJPEG USB Camera
^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- USB camera MJPEG stream capture
- Low-latency compressed video streaming
- Hardware acceleration support

**Auxiliary Features**:

- NVJPEG hardware acceleration on Jetson
- Software/hardware-assisted decoding on X86
- Configurable camera parameters

**Adapters Used**:

- :ref:`FFmpeg source adapter <ffmpeg_source_adapter>` (USB camera)
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `MJPEG USB Camera <https://github.com/insight-platform/Savant/tree/develop/samples/mjpeg_usb_cam>`_

Multiple RTSP Streams
^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Multiple RTSP stream ingestion
- Multiplexed stream processing
- PeopleNet model processing on multiple streams

**Auxiliary Features**:

- Stream multiplexing demonstration
- Multi-source processing pipeline
- LL-HLS output support

**Adapters Used**:

- :ref:`RTSP source adapter <ffmpeg_rtsp_source_adapter>` (multiple instances)
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Multiple RTSP <https://github.com/insight-platform/Savant/tree/develop/samples/multiple_rtsp>`_

Multiple GigE Vision Cameras
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- GigE Vision camera integration
- Multiple camera stream processing
- Raw RGBA and HEVC-encoded frame support
- GigE Vision Source Adapter demonstration

**Auxiliary Features**:

- Stream control API
- Multi-format camera support
- Real-time multi-camera processing

**Adapters Used**:

- :ref:`GigE camera source adapter <gige_vision_source_adapter>` (multiple instances)
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Multiple GigE <https://github.com/insight-platform/Savant/tree/develop/samples/multiple_gige>`_

RTSP Camera Compatibility Test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- RTSP camera compatibility testing
- FFmpeg and Retina RTSP adapter variants
- NVDEC and NVENC integration testing
- Camera compatibility validation

**Auxiliary Features**:

- RTCP Sender Reports support (Retina adapter)
- Cross-stream synchronization testing
- Multiple adapter variant testing

**Adapters Used**:

- :ref:`FFmpeg source adapter <ffmpeg_source_adapter>` OR Retina RTSP source adapter
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `RTSP Camera Compatibility Test <https://github.com/insight-platform/Savant/tree/develop/samples/rtsp_cam_compatibility_test>`_

AWS Kinesis Integration
^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Kinesis Video Stream integration
- Frame export/import pipeline
- MongoDB metadata storage

**Auxiliary Features**:

- Cloud streaming capabilities
- Metadata synchronization
- AWS service integration

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Kinesis Video Stream sink adapter <multistream_kinesis_video_stream_sink_adapter>`
- :ref:`Kinesis Video Stream source adapter <kinesis_video_stream_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `AWS Kinesis <https://github.com/insight-platform/Savant/tree/develop/samples/aws_kinesis>`_

Processing Control and Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditional Video Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Conditional processing based on Etcd parameters
- Dynamic processing enable/disable
- PeopleNet-based conditional encoding

**Auxiliary Features**:

- Etcd-based source control
- Tag-based processing pipeline
- DrawFunc and encoder control

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Conditional Video Processing <https://github.com/insight-platform/Savant/tree/develop/samples/conditional_video_processing>`_

Buffer Adapter Demo
^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Buffer adapter functionality demonstration
- Load spike simulation and handling
- Frame buffering and dropping strategies

**Auxiliary Features**:

- Prometheus metrics integration
- Grafana dashboard visualization
- Performance monitoring

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Buffer adapter <buffer_bridge_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Buffer Adapter <https://github.com/insight-platform/Savant/tree/develop/samples/buffer_adapter>`_

Auxiliary Streams
^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Multiple resolution stream generation
- Auxiliary stream demonstration
- Frame scaling to different resolutions

**Auxiliary Features**:

- Multi-resolution output (360p, 480p, 720p)
- Encoder optimization
- Upscaling demonstration

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>` (multi-stream)

**Link**: `Auxiliary Streams <https://github.com/insight-platform/Savant/tree/develop/samples/auxiliary_streams>`_

PyGroup: Colocated PyFuncs
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Multiple sequential PyFuncs colocated in a single ``pygroup`` element
- Sequential per-frame execution without inter-element queues
- Per-stage OpenTelemetry spans preserved for each colocated PyFunc

**Auxiliary Features**:

- Overlay drawing on the main frame
- Per-stage auxiliary stream generation
- Preconfigured Jaeger/OTLP tracing

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>` (multi-stream)

**Link**: `PyGroup <https://github.com/insight-platform/Savant/tree/develop/samples/pygroup>`_

Data Integration and APIs
-------------------------

Kafka-Redis Adapter
~~~~~~~~~~~~~~~~~~~

**Platform Support**: X86 + L4T

**Main Features**:

- Kafka-Redis adapter demonstration
- Frame content storage in KeyDB/Redis
- Metadata storage in Kafka
- Pub-sub architecture implementation

**Auxiliary Features**:

- KeyDB alternative to Redis
- Message broker integration
- Distributed processing support

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Kafka-Redis sink adapter <kafka_redis_sink_adapter>`
- :ref:`Kafka-Redis source adapter <kafka_redis_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Kafka-Redis Adapter <https://github.com/insight-platform/Savant/tree/develop/samples/kafka_redis_adapter>`_

Key-Value API
~~~~~~~~~~~~~

**Platform Support**: X86 only

**Main Features**:

- Embedded Key-Value store demonstration
- REST API access to KV store
- WebSocket subscription support

**Auxiliary Features**:

- HTTP API integration
- Real-time data subscription
- Protobuf serialization support

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`

**Link**: `Key-Value API <https://github.com/insight-platform/Savant/tree/develop/samples/key_value_api>`_

Source Adapter with JSON Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Platform Support**: X86 + L4T

**Main Features**:

- JSON metadata integration with video sources
- External metadata injection
- Metadata-video synchronization

**Auxiliary Features**:

- Custom metadata handling
- JSON format support
- Synchronized processing

**Adapters Used**:

- Media files source adapter (with JSON metadata) - see :ref:`Image File Source Adapter <image_file_source_adapter>` or :ref:`Video File Source Adapter <video_file_source_adapter>`
- :ref:`Image files sink adapter <image_file_sink_adapter>`

**Link**: `Source Adapter with JSON Metadata <https://github.com/insight-platform/Savant/tree/develop/samples/source_adapter_with_json_metadata>`_

Development and Testing Tools
-----------------------------

Template Sample
~~~~~~~~~~~~~~~

**Platform Support**: X86 + L4T

**Main Features**:

- Complete development template for custom Savant modules
- Basic pipeline with dev features
- Client SDK demonstration

**Auxiliary Features**:

- Docker Compose and devcontainer setup
- Jaeger tracing integration
- PyFunc and DrawFunc templates
- Hot-reload development workflow

**Adapters Used**:

- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`
- URI Input script integration

**Link**: `Template <https://github.com/insight-platform/Savant/tree/develop/samples/template>`_

Bypass Model
~~~~~~~~~~~~

**Platform Support**: X86 + L4T

**Main Features**:

- Identity model for preprocessing troubleshooting
- Data preprocessing demonstration
- Model input/output comparison

**Auxiliary Features**:

- PyTorch to ONNX conversion
- Aspect ratio maintenance
- Preprocessing visualization

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Bypass Model <https://github.com/insight-platform/Savant/tree/develop/samples/bypass_model>`_

Pipeline Watchdog
~~~~~~~~~~~~~~~~~

**Platform Support**: X86 only

**Main Features**:

- Pipeline monitoring and watchdog functionality
- Random processing delays simulation
- Pipeline health monitoring

**Auxiliary Features**:

- Configurable delay parameters
- Sink monitoring capabilities
- Reliability testing

**Adapters Used**:

- Custom sink monitoring via Client SDK

**Link**: `Pipeline Watchdog <https://github.com/insight-platform/Savant/tree/develop/samples/pipeline_watchdog>`_

Telemetry and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

OpenTelemetry Example
^^^^^^^^^^^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- OpenTelemetry integration demonstration
- Distributed tracing with Jaeger
- Performance monitoring and debugging

**Auxiliary Features**:

- TLS support for telemetry collection
- Span instrumentation examples
- Error tracking and visualization
- Custom telemetry collection

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Telemetry <https://github.com/insight-platform/Savant/tree/develop/samples/telemetry>`_

Router Demo
^^^^^^^^^^^

**Platform Support**: X86 + L4T

**Main Features**:

- Stream routing demonstration
- Multi-sink pipeline routing
- Keyframe-based routing logic

**Auxiliary Features**:

- Screenshot generation
- Video archiving
- Frame filtering

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Video files sink adapter <video_file_sink_adapter>`

**Link**: `Router <https://github.com/insight-platform/Savant/tree/develop/samples/router>`_

Specialized Processing
----------------------

Pass-Through Processing
~~~~~~~~~~~~~~~~~~~~~~~

**Platform Support**: X86 + L4T

**Main Features**:

- Minimal processing pipeline demonstration
- Frame pass-through without modification
- Pipeline overhead measurement

**Auxiliary Features**:

- Performance baseline establishment
- Minimal latency processing
- Throughput optimization

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>`
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Pass-Through Processing <https://github.com/insight-platform/Savant/tree/develop/samples/pass_through_processing>`_

Original Resolution Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Platform Support**: X86 + L4T

**Main Features**:

- Processing at original video resolution
- No scaling or resolution changes
- Native resolution handling

**Auxiliary Features**:

- Resolution preservation
- Quality maintenance
- Performance optimization

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>` (multiple instances for different resolutions)
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Original Resolution Processing <https://github.com/insight-platform/Savant/tree/develop/samples/original_resolution_processing>`_

Source Shaper Sample
~~~~~~~~~~~~~~~~~~~~

**Platform Support**: X86 + L4T

**Main Features**:

- Video source shaping and preprocessing
- Frame rate and resolution control
- Source adaptation demonstration

**Auxiliary Features**:

- Dynamic source modification
- Frame rate adjustment
- Resolution scaling

**Adapters Used**:

- :ref:`Video loop source adapter <video_loop_source_adapter>` (multiple instances for different sources)
- :ref:`Always On RTSP sink adapter <always_on_rtsp_sink_adapter>`

**Link**: `Source Shaper Sample <https://github.com/insight-platform/Savant/tree/develop/samples/source_shaper_sample>`_

Getting Started
---------------

To explore any of these samples:

1. **Prerequisites**: Ensure your environment is properly configured by running::

    git clone https://github.com/insight-platform/Savant.git
    cd Savant
    git lfs pull
    ./utils/check-environment-compatible

2. Visit the sample page for detailed instructions and configuration options.

Performance Notes
-----------------

- **First-time execution**: Many samples require model compilation to TensorRT engines, which can take 10-40 minutes depending on the model complexity
- **Platform optimization**: Samples are tested for their supported platforms with appropriate hardware acceleration

For detailed information about each sample, including specific setup instructions and configuration options, please visit the individual sample links provided above.

