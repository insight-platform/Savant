# Savant: High-Performance Computer Vision Framework For Data Center And Edge

![GitHub release (with filter)](https://img.shields.io/github/v/release/insight-platform/Savant)
[![Build status](https://github.com/insight-platform/Savant/actions/workflows/main.yml/badge.svg?branch=develop)](https://github.com/insight-platform/Savant/actions/workflows/main.yml)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/SavantFramework.svg?style=social&label=Follow%20%40SavantFramework)](https://twitter.com/SavantFramework) [![Blog](https://img.shields.io/badge/Inside%20InSight%20Blog-444444?logo=medium)](https://b.savant-ai.io/) [![Discord](https://img.shields.io/badge/Discord-8A2BE2)](https://discord.gg/KVAfGBsZGd)

:star: Star us on GitHub — it motivates us a lot and helps the project become more visible to developers.

![current-demos-page](https://github.com/insight-platform/Savant/assets/15047882/22a58429-582f-4e46-a5dd-74c54e863bec)

Savant is an open-source, high-level framework for building real-time, streaming, highly efficient multimedia AI
applications on the Nvidia stack. It helps to develop dynamic, fault-tolerant inference pipelines that utilize the best
Nvidia approaches for data center and edge accelerators.

Savant is built on DeepStream and provides a high-level abstraction layer for building inference pipelines. It is
designed to be easy to use, flexible, and scalable. It is a great choice for building both real-time or high-load
computer vision and video analytics applications.

## What Version To Use

Savant depends on Nvidia DeepStream and JetPack versions (Jetson). The following tables show the compatibility of Savant
versions with DeepStream versions.

### 0.2.11 LTS

This release is recommended for production use. It uses the time-proven DeepStream 6.3. The release works on dGPU (
Turing, Volta, Ampere, Ada) and Jetson (Xavier NX/AGX, Orin Nano/NX/AGX) hardware.

Known drawbacks:

- NVJPEG caps on 115MHz on Jetson Orin Nano in JPEG decoding.

| Requirements                                    | Status | DeepStream |
|-------------------------------------------------|--------|------------|
| X86 Driver 525(Datacenter), 530+ Quadro/GeForce | Stable | 6.3        |
| Jetson Xavier, Orin with JetPack 5.1.2 GA       | Stable | 6.3        |           
   
### 0.4.x Current Production Release

This release contains new features and is tested for production use. It is a choice for users requiring functionality missing in 0.2.x and 0.3.x.

| Requirements                                    | Status | DeepStream |
|-------------------------------------------------|--------|------------|
| X86 Driver 525(Datacenter), 530+ Quadro/GeForce | Stable | 6.4        |
| Jetson Orin JetPack 6.0                         | Stable | 6.4        |

### 0.5.x Current Development

This release integrates DeepStream 7.0

| Requirements                                    | Status | DeepStream |
|-------------------------------------------------|--------|------------|
| X86 Driver 525(Datacenter), 530+ Quadro/GeForce | Stable | 7.0        |
| Jetson Orin JetPack 6.0                         | Stable | 7.0        |

## Chat With Us

The best way to approach us is [Discord](https://discord.gg/KVAfGBsZGd). We are always happy to help you with any
questions you may have.

## Quick Links

- [Blog](https://b.savant-ai.io/)
- [Getting Started Tutorial](https://docs.savant-ai.io/develop/getting_started/2_module_devguide.html)
- [Pipeline Samples](https://github.com/insight-platform/Savant/tree/develop/samples)
- [Documentation](https://docs.savant-ai.io/)
- [Performance Regression Tracking Dashboard](docs/performance.md)

## Getting Started

First, take a look at the runtime
configuration [guide](https://docs.savant-ai.io/develop/getting_started/0_configure_prod_env.html) to configure the
working environment.

The [demo](https://github.com/insight-platform/Savant/tree/develop/samples/peoplenet_detector) shows a pipeline
featuring person detection, facial detection, tracking, facial blurring (OpenCV CUDA), and a real-time analytics
dashboard:

![](samples/peoplenet_detector/assets/peoplenet-blur-demo-loop-400.webp)

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/peoplenet_detector
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/city-traffic' in your player
# or visit 'http://127.0.0.1:888/stream/city-traffic/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

## Who Would Be Interested in Savant

If your task is to implement high-performance production-ready computer vision and video analytics applications, Savant
is for you.

With Savant, developers:

- get the maximum performance on Nvidia equipment on edge and in the core;
- decrease time to market when building dynamic pipelines with DeepStream technology but without low-level programming;
- develop easily maintainable and testable applications with a well-established framework API;
- build heterogeneous pipelines with different models and data sources;
- build hybrid edge/datacenter applications with the same codebase;
- monitor and trace the pipelines with OpenTelemetry and Prometheus;
- implement on-demand and non-linear processing by utilizing [Replay](https://github.com/insight-platform/Replay).

## Runs On Nvidia Hardware

Savant components, processing video and computer vision, require Nvidia hardware. We support the following devices:

- Jetson Xavier NX/AGX (0.2.x);
- Jetson Orin Nano/NX/AGX (0.3.x and newer);
- Nvidia Turing, Ampere, Ada, Hopper, Blackwell GPUs (0.2.x and newer).

## Why We Developed Savant

We developed Savant give computer vision and video analytics engineers a ready-to-use stack for building real-life
computer vision applications working at the edge and in the data center. Unlike other computer vision frameworks
like PyTorch, TensorFlow, OpenVINO/DlStreamer, and DeepStream, Savant provides users with not only inference and image
manipulation tools but also advanced architecture for building distributed edge/datacenter computer vision applications
communicating over the network. Thus, Savant users focus on computer vision but do not reinvent the wheel, when
developing their applications.

Savant is a very high-level framework hiding low-level internals from developers: computer vision pipelines consist of
declarative (YAML) blocks with Python functions.

## Features

Savant is packed with many features skyrocketing the development of high-performing computer vision applications.

### 🔧 All You Need for Building Real-Life Applications

Savant supports everything you need for developing advanced pipelines: detection, classification, segmentation,
tracking, and custom pre- and post-processing for meta and images.

We have implemented samples demonstrating pipelines you can build with Savant. Visit the [samples](samples) folder to
learn more.

### 🚀 High Performance

Savant is designed to be fast: it works on top of DeepStream - the fastest SDK for video analytics. Even the heavyweight
segmentation models can run in real-time on Savant. See
the [Performance Regression Tracking Dashboard](docs/performance.md) for the latest performance results.

### 🌐 Works On Edge and Data Center Equipment

The framework supports running the pipelines on both Nvidia's edge devices (Jetson Family) and data center devices (
Tesla, Quadro, etc.) with minor or zero changes.

### ❤️ Cloud-Ready

Savant pipelines run in Docker containers. We provide images for x86+dGPU and Jetson hardware. Integrated OpenTelemetry
and Prometheus support enable monitoring and tracing of the pipelines.

### ⚡ Low Latency and High Capacity Processing

Savant can be configured to execute a pipeline in real-time, skipping data when running out of capacity or in high
capacity mode, which guarantees the processing of all the data, maximizing the utilization of the available resources.

### 🤝 Ready-To-Use API

A pipeline is a self-sufficient service communicating with the world via high-performance streaming API. Whether
developers use provided adapters or Client SDK, both approaches use the API.

### 📁 Advanced Data Protocol

The framework universally uses a common protocol for both video and metadata delivery. The protocol is highly flexible,
allowing video-related information alongside arbitrary structures useful for IoT and 3rd-party integrations.

### ⏱ OpenTelemetry Support

In Savant, you can precisely instrument pipelines with OpenTelemetry: a unified monitoring solution. You can use sampled
or complete traces to balance the performance and precision. The traces can span from edge to core to business logic
through network and storage because their propagation is supported by the Savant protocol.

### 📊 Prometheus Support

Savant pipelines can be instrumented with Prometheus: a popular monitoring solution. Prometheus is a great choice for
monitoring the pipeline's performance and resource utilization.

### 🧰 Client SDK

We provide Python-based SDK to interact with Savant pipelines (ingest and receive data). It enables simple integration
with 3rd-party services. Client SDK is integrated with OpenTelemetry providing programmatic access to the pipeline
traces and logs.

### 🧘 Development Server

Software development for vanilla DeepStream is a pain. Savant provides a Development Server tool, which enables dynamic
reloading of changed code without pipeline restarts. It helps to develop and debug pipelines much faster. Altogether
with Client SDK, it makes the development of DeepStream-enabled applications really simple. With the Development Server,
you can develop remotely on a Jetson device or server right from your IDE.

### 🔀 Dynamic Sources Management

In Savant, you can dynamically attach and detach sources and sinks to the pipeline without reloading. The framework
resiliently handles situations related to source/sink outages.

### 🏹 Handy Source and Sink Adapters

The communication interface is not limited to Client SDK: we provide several ready-to-use adapters, which you can use "
as is" or modify for your needs.

The following source adapters are available:

- [Local video file](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#video-file-source-adapter);
- [Local directory of video files](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#video-file-source-adapter);
- [Video URL](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#video-file-source-adapter);
- [Local image file](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#image-file-source-adapter);
- [Local directory of image files](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#image-file-source-adapter);
- [Image URL](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#image-file-source-adapter);
- [RTSP stream](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#rtsp-source-adapter);
- [USB/CSI camera](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#usb-cam-source-adapter);
- [GigE (Genicam) industrial cam](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#gige-vision-source-adapter);
- [Kafka-Redis](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#kafka-redis-source-adapter);
- [Video loop URL](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#video-loop-source-adapter);
- [Multi-stream Source](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#multi-stream-source-adapter);
- [Amazon Kinesis Video Streams Source](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#kinesis-video-stream-source-adapter);
- [Message Dump Player](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#message-dump-player-source-adapter).

Several sink adapters are implemented:

- [Inference results are placed into JSON file stream](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#json-metadata-sink-adapter);
- [Resulting video overlay displayed on a screen (per source)](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#display-sink-adapter);
- [MP4 file (per source)](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#video-file-sink-adapter);
- [Image directory (per source)](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#image-file-sink-adapter);
- [Always-On RTSP Stream Sink](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#always-on-rtsp-sink-adapter);
- [Kafka-Redis](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#kafka-redis-sink-adapter);
- [Amazon Kinesis Video Streams Sink](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#multistream-kinesis-video-stream-sink-adapter).

### 🎯 Dynamic Parameters Ingestion

Advanced ML pipelines may require information from the external environment for their work. The framework enables
dynamic configuration of the pipeline with:

- ingested frame attributes passed in per-frame metadata;
- Etcd's attributes watched and instantly applied;
- 3rd-party attributes, which are received through user-defined functions.

### 🖼 OpenCV CUDA Support

Savant supports custom OpenCV CUDA bindings enabling operations on DeepStream's in-GPU frames with a broad range of
OpenCV CUDA functions: the feature helps in implementing highly efficient video transformations, including but not
limited to blurring, cropping, clipping, applying banners and graphical elements over the frame, and others. The feature
is available from Python.

### 🔦 PyTorch Support

Savant supports PyTorch, one of the most popular ML frameworks. It enables the developer to use ready-to-use PyTorch
models from PyTorchHub, a large number of code samples, and reliable extensions. The integration is highly efficient: it
allows running inference on GPU-allocated images and processing the results in GPU RAM, avoiding data transfers between
CPU and GPU RAM.

### 🔢 CuPy Support For Post-Processing

Savant supports CuPy: a NumPy-like library for GPU-accelerated computing. It enables the developer to implement custom
post-processing functions in Python, executed in GPU RAM, avoiding data transfers between CPU and GPU RAM. The feature
allows for accessing model output tensors directly from GPU RAM, which helps implement heavy-weight custom
post-processing functions.

The integration also provides a conversion for in-GPU data between CuPy, OpenCV, and PyTorch in-GPU formats.

### ↻ Rotated Detection Models Support

We frequently deal with the models resulting in bounding boxes rotated relative to a video frame (oriented bounding
boxes). For example, it is often the case with bird-eye cameras observing the underlying area from a high point.

Such cases may require detecting the objects with minimal overlap. To achieve that, special models are used which
generate bounding boxes that are not orthogonal to the frame axis. Take a look
at [RAPiD](https://vip.bu.edu/projects/vsns/cossy/fisheye/rapid/) to find more.

[![image](https://user-images.githubusercontent.com/15047882/167245173-aa0a18cd-06c9-4517-8817-253d120c0e07.png)](#)

### ⇶ Parallelization

Savant supports processing parallelization; it helps to utilize the available resources to the maximum. The
parallelization is achieved by running the pipeline stages in separate threads. Despite flow control-related Python code
is not parallel; the developer can utilize GIL-releasing mechanisms to achieve the desired parallelization with NumPy,
Numba, or custom native code in C++ or Rust.

## What's Next

- [Getting Started Tutorial](https://docs.savant-ai.io/develop/getting_started/2_module_devguide.html)
- [Publications and Samples](https://github.com/insight-platform/Savant/tree/develop/samples)
- [Documentation](https://docs.savant-ai.io/)

## Contribution

We welcome anyone who wishes to contribute, report, and learn.

## About Us

The In-Sight team is an ML/AI department of [Bitworks Software](https://bitworks.software/). We develop custom high 
performance CV applications for various industries providing full-cycle process, which includes but not limited to data 
labeling, model evaluation, training, pruning, quantization, validation, and verification, pipelines development, CI/CD. 
We are mostly focused on Nvidia hardware (both datacenter and edge).

Contact us: info@bw-sw.com
