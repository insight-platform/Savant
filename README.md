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
designed to be easy to use, flexible, and scalable. It is a great choice for building smart CV and video analytics
applications for cities, retail, manufacturing, and more. Savant is a member of the Nvidia Inception Program.

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

### 0.3.11 LTS

This release is recommended for production use with DeepStream 6.4. The release works on dGPU (Turing, Volta, Ampere,
Ada) and Jetson Orin Nano/NX/AGX hardware (JetPack 6.0 DP). It does not support Jetson Xavier and older devices.

| Requirements                                    | Status | DeepStream |
|-------------------------------------------------|--------|------------|
| X86 Driver 525(Datacenter), 530+ Quadro/GeForce | Stable | 6.4        |
| Jetson Orin JetPack 6.0 DP                      | Stable | 6.4        |           

### 0.4.x Current Develop (Feature Releases)

This branch represents current development. It is not recommended for production use. It is a good choice for testing
new features and providing feedback and also if you require new features absent in `0.2.x` and `0.3.x`.

| Requirements                                    | Status | DeepStream |
|-------------------------------------------------|--------|------------|
| X86 Driver 525(Datacenter), 530+ Quadro/GeForce | Stable | 6.4        |
| Jetson Orin JetPack 6.0 DP                      | Stable | 6.4        |

## Chat With Us

The best way to approach us is [Discord](https://discord.gg/KVAfGBsZGd). We are always happy to help you with any
questions you may have.

## Quick Links

- [Blog](https://b.savant-ai.io/)
- [Getting Started Tutorial](https://docs.savant-ai.io/develop/getting_started/2_module_devguide.html)
- [Pipeline Samples](https://github.com/insight-platform/Savant/tree/develop/samples)
- [Documentation](https://docs.savant-ai.io/)
- [Performance Regression Tracking Dashboard](docs/performance.md)

## Quick Start

Runtime configuration [guide](https://docs.savant-ai.io/develop/getting_started/0_configure_prod_env.html) helps to
configure the runtime to run Savant pipelines.

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

## What Savant Is Not

Savant is not for AI model training; it's for building fast streaming inference applications working on Edge and Core
Nvidia equipment. We use PyTorch to train our models and recommend sticking with it.

## Who Would Be Interested in Savant?

If your task is to implement high-performance production-ready computer vision and video analytics applications, Savant
is for you. It helps to:

- get the maximum performance on Nvidia equipment on edge and in the core;
- decrease time to market when building dynamic pipelines with DeepStream technology but without low-level programming;
- develop easily maintainable and testable applications with a well-established framework API.

## Runs On Nvidia Hardware

- Nvidia Jetson NX/AGX, Orin Nano/NX/AGX;
- Nvidia Turing GPU;
- Nvidia Ampere GPU;
- Nvidia Hopper, hopefully, we did not have a chance to try it yet :-)

## About Nvidia DeepStream

[Nvidia DeepStream](https://developer.nvidia.com/deepstream-sdk) is today's most advanced toolkit for developing
high-performance real-time computer vision AI applications that run magnitude times faster than conventional AI
applications executed within the runtimes like PyTorch, TensorFlow and similar.

[![Nvidia DeepStream Picture](https://developer.nvidia.com/sites/default/files/akamai/deepstream/metropolis-and-iva-deepstreadm-sdk-block-diagrams-2009801-r1-1.png)](#)

The top-notch performance is achieved by specially designed software using the best Nvidia accelerator features,
including hardware encoding and decoding for video streams, moving the frames through inference blocks solely in GPU RAM
without data transfers into CPU RAM and back. The inference blocks utilize the highly efficient
low-level ([TensorRT](https://developer.nvidia.com/tensorrt)) software stack, optimizing inference operations to get the
best of the hardware used.

## Why We Developed Savant?

Why do we develop Savant if DeepStream solves the problem? That is because DeepStream is a challenging-to-use
technology; it does not define software architecture, just a bunch of plug-ins for GStreamer: the open-source multimedia
framework for building highly-efficient streaming applications.

It makes developing more or less sophisticated DeepStream applications very painful because the developer must
understand how the GStreamer processes the data, making the learning curve steep and almost unreachable for ML engineers
focused on model training.

Savant is a very high-level framework on DeepStream, hiding low-level internals from the developer and providing
practical tools for quickly implementing real-life streaming AI applications. So, you implement your inference pipeline
as a set of declarative (YAML) blocks with several user-defined functions in Python (or C/C++ if you would like to
utilize most of the CUDA runtime).

## Features

Savant is packed with several killer features which skyrocket the development of Deepstream applications.

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
- [Amazon Kinesis Video Streams Source](https://docs.savant-ai.io/develop/savant_101/10_adapters.html#kinesis-video-stream-source-adapter).

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

The In-Sight team is a ML/AI department of Bitworks Software. We develop custom high performance CV applications for
various industries providing full-cycle process, which includes but not limited to data labeling, model evaluation,
training, pruning, quantization, validation, and verification, pipelines development, CI/CD. We are mostly focused on
Nvidia hardware (both datacenter and edge).

Contact us: info@bw-sw.com
