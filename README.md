# Savant

Savant is an open-source, high-level framework for building real-time, streaming, highly efficient multimedia AI applications on the Nvidia stack. It makes it possible to develop the dynamic, fault-tolerant inference pipelines which utilize the best Nvidia approaches for both data center and edge accelerators with a very moderate amount of custom code in Python.

## TL;DR Links

- [Getting Started]()
- [Publications and Samples]()
- [API Documentation]()

## About DeepStream 

[Nvidia DeepStream](https://developer.nvidia.com/deepstream-sdk) is today's most advanced toolkit for developing high-performance real-time streaming AI applications that run several times faster than conventional AI applications executed within the conventional runtimes like PyTorch, TensorFlow and similar. 

[![Nvidia DeepStream Picture](https://developer.nvidia.com/sites/default/files/akamai/deepstream/metropolis-and-iva-deepstreadm-sdk-block-diagrams-2009801-r1-1.png)](#)

That is possible due to specially designed architecture, which utilizes the best Nvidia accelerator features, including hardware encoding and decoding for video streams, moving the frames thru inference blocks mostly in GPU RAM without excessive data transfers between CPU and GPU RAM. It also stands on a highly optimized low-level software stack that optimizes inference operations to get the best of the hardware used.

## Why?

Why choose Savant if DeepStream solves the problem? Because DeepStream is a very tough and challenging to use technology.

The root of that is that DeepStream is implemented as a set of plug-ins for Gstreamer - the open-source multi-media framework for highly-efficient streaming applications. It makes developing more or less sophisticated DeepStream applications very difficult because the developer must understand how the Gstreamer processes the data. That makes the learning curve very steep.

Savant is a very high-level framework over the DeepStream, which incapsulates all the Gstreamer internals from the developer and provides a beneficial tools for implementing real-life streaming AI inference applications. So, as a developer, you implement your inference pipeline as a set of declarative (YAML) blocks with several user-defined functions in Python (and C/C++ if you would like to utilize the most of CUDA runtime).

The Savant framework provides many really needed features out of the box. When we created Savant, we utilized the best of our experience, which we achieved through several years of commercial DeepStream development.

## Features

### Dynamic Inference Variables

The framework gives the developer the means to configure and reconfigure the inference pipeline operational parameters dynamically with:
- incoming per-frame metadata;
- Etcd variables which a watched and instantly applied;
- 3rd-party sources that are accessible through user-defined functions.

### Dynamic Sources Management

Dynamic Sources Management
Savant allows the developer to handle virtual streams of anything without restarting the pipeline. The video files, sets of video files, image collections, network video streams, and raw video frames (USB, GigE) - all is processed universally without the need to reload the pipeline to attach or detach the stream. 

The framework virtualizes the stream concept by decoupling it from the real-life data source and takes care of a garbage collection for no longer available streams.

As a developer, you use handy source adapters to ingest media data into the framework runtime and use sink adapters to get the results out of it. The adapters can transfer the media through the network or locally. We have already implemented plenty of them useful in a real-life, and you can implement the required one for you if needed - the protocol is simple and utilizes standard open source tools.

The decoupled nature of adapters also provides better reliability because the failed data source affects the adapter operation, not a framework operation.

### Rotated Bounding Boxes Out Of The Box

In our practice, when we create commercial AI software, we often meet the cases where the bounding boxes rotated relative to a video frame. For example, it is often the case when the camera observes the viewport from the ceiling when the objects reside on the floor. 

These cases require detecting the objects in the way the parts of other things hit the bounding box minimally. To achieve that, the developers use unique models that calculate box angle like [RAPiD](https://vip.bu.edu/projects/vsns/cossy/fisheye/rapid/).

[![image](https://user-images.githubusercontent.com/15047882/167245173-aa0a18cd-06c9-4517-8817-253d120c0e07.png)](#)

Such models require additional post-processing, which involves the rotation because otherwise, you cannot utilize most of the classifier models as they need orthogonal boxes as their input.

Savant supports the bounding box rotation out of the box right before passing them to the classifier models.

### Works On Edge and Data Center Equipment

The framework is designed and developed in such a way to run the pipelines on both edge Nvidia devices (Jetson Family) and datacenter devices (like Tesla, Quadro, etc.) with minor or zero changes.

Despite the enormous efforts of Nvidia to make the devices fully compatible, there are architectural features that require special processing to make the code compatible between discrete GPU and Jetson appliances.

Even DeepStream itself sometimes behaves unpredictably in certain conditions. The framework code handles those corner cases to avoid crashes or misbehavior.

### Low Latency and Capacity Processing

When running an inference application on an edge device, the developer usually wants real-time performance. Such requirement is due to the nature of the edge - the users place devices near the live data sources like sensors or video cameras, and they expect the device capacity is enough to handle incoming messages or video-stream without the loss. 

Edge devices usually are low in computing resources, including the storage, CPU, GPU, and RAM, so their overuse is not desired because it could lead to data loss.

On the other hand, the data transmitted to the data center are expected to be processed with latency and delay (because the transmission itself introduces that delay and latency). 

Servers deployed in the data center have many resources - dozens of cores, lots of RAM, very powerful GPU accelerators, and a large amount of storage. It makes it possible to run capacity processing ingesting the data to devices from the files or message brokers (like Apache Kafka) to utilize 100% of the device, limiting the rate only by backpressure of the processing pipeline. Also, the data center devices process the data in parallel - by increasing the number of GPU accelerators installed in the server and among servers.

Savant provides the configuration means to run the pipeline in a real-time mode, which skips the data if the device is incapable of handling them in the real-time, and in synchronous mode, which guarantees the processing of all the data in a capacity way, maximizing the utilization of the available resources.

### Handy Source and Sink Adapters

In DeepStream, the sources and sinks are part of the Gstreamer pipeline because it's by design. However, such a design makes it difficult to create reliable applications in the real world. 

There are reasons for that. The first one is low reliability. The source and sink are external entities that, being coupled into the pipeline, can make it crash when they are failed or are no longer available. E.g., when the RTSP camera is not available, the corresponding RTSP Gstreamer source signals the pipeline to terminate. 

The problem becomes dramatic when multiple sources ingest data into a single pipeline - a natural case in the real world. You don't want to load multiple instances of the same AI models into the GPU because of RAM limitations and overall resource overutilization. So, following the natural Gstreamer approach, you have a muxer scheme with a high chance of failing if any source fails. 

That's why you want to have sources decoupled from the pipeline - to increase the stability of the pipeline and avoid unnecessarily reloads in case of source failure.

Another reason is dynamic source management which is a very difficult task when managed through the Gstreamer directly. You have to implement the logic which attaches and detaches the sources and sinks when needed.

The third reason is connected with media formats. You have to reconfigure Gstremer pads setting proper capabilities when the data source changes the format of media, e.g., switching from h.264 to HEVC codec. The simplest way to do that is to crash and recover, which causes significant unavailability time while AI models are compiled to TensorRT and loaded in GPU RAM. So, you want to avoid that as well.

Savant introduces all the means which solve all the mentioned problems magically without the need to handle them someway.

We have implemented several ready-to-use adapters which you can utilize instantly or use as a foundation to develop your own.

Currently, the following source adapters are available:
- Local video file source;
- URL video source;
- Local directory of video files;
- RTSP source;
- Local image file source;
- URL Image source;
- Image directory source;
- USB cam source;

There are basic sink adapters are implemented:
- Inference results to JSON file stream;
- Resulting video overlay displayed on a screen (per source);
- MP4 file (per source).

The framework uses an established protocol based on Apache AVRO, both for sources and sinks. The sources and sinks talk to Savant through ZeroMQ sockets.

### Easy to Deploy
