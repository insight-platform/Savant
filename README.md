# Savant: Supercharged Computer Vision and Video Analytics Framework on DeepStream


Savant is an open-source, high-level framework for building real-time, streaming, highly efficient multimedia AI applications on the Nvidia stack. It helps to develop dynamic, fault-tolerant inference pipelines that utilize the best Nvidia approaches for data center and edge accelerators.

Savant is built on DeepStream and provides a high-level abstraction layer for building inference pipelines. It is designed to be easy to use, flexible, and scalable. It is a great choice for building smart CV and video analytics applications for cities, retail, manufacturing, and more.

:star: Star us on GitHub — it motivates us a lot and helps the project become more visible to developers.

![GitHub release (with filter)](https://img.shields.io/github/v/release/insight-platform/Savant)
[![Build status](https://github.com/insight-platform/Savant/actions/workflows/build-docker-latest.yml/badge.svg?branch=develop)](https://github.com/insight-platform/Savant/actions/workflows/build-docker-latest.yml)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/SavantFramework.svg?style=social&label=Follow%20%40SavantFramework)](https://twitter.com/SavantFramework) [![Blog](https://img.shields.io/badge/Inside%20InSight%20Blog-444444?logo=medium)](https://blog.savant-ai.io/)

Savant is a member of Nvidia Inception Program:

![savant_inception_member](https://github.com/insight-platform/Savant/assets/15047882/12928291-05cc-4639-b43c-6b13d36a01fd)

## Chat With Us

The best way to approach the Savant team is to join our Discord server. We are always happy to help you with any questions you may have.

[![discord](https://user-images.githubusercontent.com/15047882/229273271-d033e597-06d3-4aeb-b93d-1217e95ca07e.png)](https://discord.gg/KVAfGBsZGd)


## Quick Links

- [Blog](https://blog.savant-ai.io/)
- [Getting Started Tutorial](https://blog.savant-ai.io/meet-savant-a-new-high-performance-python-video-analytics-framework-for-nvidia-hardware-22cc830ead4d?source=friends_link&sk=c7169b378b31451ab8de3d882c22a774)
- [Pipeline Samples](https://github.com/insight-platform/Savant/tree/develop/samples)
- [Documentation](https://docs.savant-ai.io/)
- [Performance Regression Tracking Dashboard](docs/performance.md)

## 1-Minute Quick Start

**Note**: Ubuntu 22.04 runtime configuration [guide](https://docs.savant-ai.io/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

Requirements:
- **X86 & Volta/Turing/Ampere/Ada Lovelace**: Linux, Drivers 525+, Docker with Compose, Nvidia Container Runtime,
- **Nvidia Jetson NX/AGX+**: JetPack 5.1+, Docker with Compose, Nvidia Container Runtime.

**Note**: Savant does not support Jetson Nano (original device released at 2020) because Nvidia doesn't support newer JetPack versions for it.

The demo shows how to make a pipeline featuring person detection, facial detection, tracking, facial blurring (OpenCV CUDA), and a real-time analytics dashboard:

![](samples/peoplenet_detector/assets/peoplenet-blur-demo-loop-400.webp)

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/peoplenet_detector
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

## What Savant Is Not

Savant is not for AI model training; it's for building fast streaming inference applications working on Edge and Core Nvidia equipment. We use PyTorch to train our models and recommend sticking with it.

## Who Would Be Interested in Savant?

If your task is to implement high-performance production-ready computer vision and video analytics applications, Savant is for you. It helps to:

- get the maximum performance on Nvidia equipment on edge and in the core;
- decrease time to market when building dynamic pipelines with DeepStream technology but without low-level programming;
- develop easily maintainable and testable applications with a well-established framework API.

## Runs On Nvidia Hardware

- Nvidia Jetson NX/AGX, Orin Nano/NX/AGX;
- Nvidia Turing GPU;
- Nvidia Ampere GPU;
- Nvidia Hopper, hopefully, we did not have a chance to try it yet :-)

## About Nvidia DeepStream

[Nvidia DeepStream](https://developer.nvidia.com/deepstream-sdk) is today's most advanced toolkit for developing high-performance real-time computer vision AI applications that run magnitude times faster than conventional AI applications executed within the runtimes like PyTorch, TensorFlow and similar.

[![Nvidia DeepStream Picture](https://developer.nvidia.com/sites/default/files/akamai/deepstream/metropolis-and-iva-deepstreadm-sdk-block-diagrams-2009801-r1-1.png)](#)

The top-notch performance is achieved by specially designed software using the best Nvidia accelerator features, including hardware encoding and decoding for video streams, moving the frames through inference blocks solely in GPU RAM without data transfers into CPU RAM and back. The inference blocks utilize the highly efficient low-level ([TensorRT](https://developer.nvidia.com/tensorrt)) software stack, optimizing inference operations to get the best of the hardware used.

## Why We Developed Savant?

Why do we develop Savant if DeepStream solves the problem? That is because DeepStream is a challenging-to-use technology; it does not define software architecture, just a bunch of plug-ins for GStreamer: the open-source multimedia framework for building highly-efficient streaming applications. 

It makes developing more or less sophisticated DeepStream applications very painful because the developer must understand how the GStreamer processes the data, making the learning curve steep and almost unreachable for ML engineers focused on model training.

Savant is a very high-level framework on DeepStream, hiding low-level internals from the developer and providing practical tools for quickly implementing real-life streaming AI applications. So, you implement your inference pipeline as a set of declarative (YAML) blocks with several user-defined functions in Python (or C/C++ if you would like to utilize most of the CUDA runtime).

## Features

Savant is packed with several killer features which skyrocket the development of Deepstream applications.

### Works On Edge and Data Center Equipment

The framework is designed and developed in such a way to run the pipelines on both edge Nvidia devices (Jetson Family) and datacenter devices (like Tesla, Quadro, etc.) with minor or zero changes.

### Easy to Deploy

The framework and the adapters are delivered as Docker images. To implement the pipeline, you take the base image, add AI models and a custom code with extra dependencies, then build the resulting image.

As for now, we provide images for x86 architecture and for Jetson hardware.

### Low Latency and Capacity Processing

Savant provides the configuration means to run the pipeline in a real-time mode, which skips the data if the device is incapable of handling them in the real-time, and in synchronous mode, which guarantees the processing of all the data in a capacity way, maximizing the utilization of the available resources.

### Ready-To-Use API

From a user perspective, the Savant pipeline is a self-contained service that accepts input data through API. It makes Savant ideal and ready for deployment within the systems. Whether developers use provided handy adapters or send data into a pipeline directly, both cases use API provided by the Savant. 

### Client SDK

We provide Python-based SDK to interact with Savant pipelines (ingest and receive data). It enables simple integration with 3rd-party services. Client SDK is integrated with OpenTelemetry enabling the access to the pipeline traces and logs.

### OpenTelemetry Support

In Savant, you can precisely trace your pipeline with OpenTelemetry, which helps to build a unified monitoring system for your infrastructure. You can use sampled or unsampled traces, which helps to balance the tracing's performance and precision. The traces can span from edge to core to business logic through network and storage.

### Development Server

Software development for vanilla DeepStream is a pain. Savant provides a Development Server tool, which enables dynamic reloading of changed code without pipeline restarts. It helps to develop and debug pipelines much faster. Altogether with Client SDK, it makes the development of DeepStream applications much more comfortable. You can even develop remotely on a Jetson device or server right from your IDE.

### Dynamic Sources Management

In Savant, you can dynamically attach and detach sources and sinks to the pipeline without reloads. The framework resiliently handles the cases when the source is not available or the sink is not available. It helps to build highly reliable applications.

### Handy Source and Sink Adapters

The communication interface is not limited to only Client SDK: we provide several ready-to-use adapters, which you can use as is or as a foundation to develop your own.

Currently, the following source adapters are available:

- [Local video file](docs/adapters.md#the-video-file-source-adapter);
- [Local directory of video files](docs/adapters.md#the-video-file-source-adapter);
- [Loop local video file](docs/adapters.md#the-video-loop-file-source-adapter);
- [Local image file](docs/adapters.md#the-image-file-source-adapter);
- [Local directory of image files](docs/adapters.md#the-image-file-source-adapter);
- [Image URL](docs/adapters.md#the-image-file-source-adapter);
- [Video URL](docs/adapters.md#the-video-file-source-adapter);
- [Video loop URL](docs/adapters.md#the-video-loop-file-source-adapter);
- [RTSP stream](docs/adapters.md#the-rtsp-source-adapter);
- [USB/CSI camera](docs/adapters.md#the-usb-cam-source-adapter);
- [GigE (Genicam) industrial cam](docs/adapters.md#the-gige-source-adapter).

There are basic sink adapters implemented:

- [Inference results are placed into JSON file stream](docs/adapters.md#the-json-meta-sink-adapter);
- [Resulting video overlay displayed on a screen (per source)](docs/adapters.md#the-display-sink-adapter);
- [MP4 file (per source)](docs/adapters.md#the-video-file-sink-adapter);
- [image directory (per source)](docs/adapters.md#the-image-file-sink-adapter);
- [Always-On RTSP Stream Sink](docs/adapters.md#the-always-on-rtsp-sink-adapter).

### Advanced Data Protocol

The framework universally uses a common protocol for both video and metadata for sources and sinks. The sources and sinks talk to Savant through ZeroMQ sockets. The protocol is highly flexible, allowing the transferring of not only video-related metadata but arbitrary structures useful for IoT and 3rd-party integrations.

### Dynamic Runtime Parameters Configuration

Sophisticated ML pipelines can use external knowledge, which helps optimize the results based on additional knowledge from the environment.

The framework enables dynamic configuration of the pipeline operational parameters with:

- ingested frame parameters passed in per-frame metadata;
- Etcd's parameters watched and instantly applied;
- 3rd-party parameters, which are received through user-defined functions.

### OpenCV CUDA Integration

Savant supports custom OpenCV CUDA bindings which allow accessing DeepStream's in-GPU frames with a broad range of OpenCV CUDA utilities: the feature helps implement highly efficient video transformations, including but not limited to blurring, cropping, clipping, applying banners and graphical elements over the frame, and others. The feature is available from Python.

### Oriented Bounding Boxes Out Of The Box

In our practice, when we develop commercial inference software, we often meet the cases where the bounding boxes rotated relative to a video frame (oriented bounding boxes). For example, it is often the case when the camera observes the viewport from the ceiling when the objects reside on the floor.

Such cases require placing the objects within boxes in a way to overlap minimally. To achieve that, we use special models that introduce bounding boxes that are not orthogonal to frame axes. Take a look at [RAPiD](https://vip.bu.edu/projects/vsns/cossy/fisheye/rapid/) to get the clue.

[![image](https://user-images.githubusercontent.com/15047882/167245173-aa0a18cd-06c9-4517-8817-253d120c0e07.png)](#)

### Parallelization

Savant provides a way to parallelize the pipeline execution. It helps to utilize the available resources to the maximum. The parallelization is achieved by running the pipeline stages in separate threads. Despite the code contains Python-based "box moving" which is not parallel; the developer can utilize GIL-releasing mechanisms to achieve the desired parallelization with NumPy, Numba, or custom native code in C++ or Rust. 

## What's Next

- [Getting Started Tutorial](https://blog.savant-ai.io/meet-savant-a-new-high-performance-python-video-analytics-framework-for-nvidia-hardware-22cc830ead4d?source=friends_link&sk=c7169b378b31451ab8de3d882c22a774)
- [Publications and Samples](https://github.com/insight-platform/Savant/tree/develop/samples)
- [Documentation](https://docs.savant-ai.io/)

## Contribution

We welcome anyone who wishes to contribute, report, and learn.

## About Us

The In-Sight team is a ML/AI department of Bitworks Software. We develop custom high performance CV applications for various industries providing full-cycle process, which includes but not limited to data labeling, model evaluation, training, pruning, quantization, validation, and verification, pipelines development, CI/CD. We are mostly focused on Nvidia hardware (both datacenter and edge).

Contact us: info@bw-sw.com
