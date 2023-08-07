# Architecture Overview

**This is a draft document**. For practically applicable knowledge, please, take a look at the [documentation](https://insight-platform.github.io/Savant/)

avant is a cutting-edge platform that leverages the power of DeepStream SDK, an advanced Nvidia framework designed for streaming AI applications. DeepStream is a cornerstone of the Nvidia ecosystem, unleashing the full potential of Nvidia accelerators in inference tasks.

DeepStream is unrivaled in its ability to deliver exceptional performance in video processing inference tasks. This is because traditional frameworks such as PyTorch, TensorFlow, OpenCV, and FFmpeg are limited by the CPU when performing operations. In contrast, DeepStream utilizes the CPU solely for flow control, while video frames are processed within the GPU memory, thereby eliminating the significant performance drop associated with transferring image frames between the CPU and GPU.

DeepStream is specifically optimized for tasks such as video decoding, frame transformations, inference, and encoding, all of which are executed within the GPU memory utilizing GPU hardware features, maximizing the potential of the accelerator card. Developing such a powerful framework requires extensive knowledge of Nvidia hardware and low-level tools such as CUDA and TensorRT, making DeepStream a highly sophisticated and advanced platform.

![Nvidia DeepStream Architecture](https://user-images.githubusercontent.com/15047882/167308102-eea0915d-e1e5-4924-bd4c-69da34d47fc7.png)

Although DeepStream offers unparalleled performance, it can be daunting for many ML engineers due to its steep learning curve and advanced knowledge domain, specifically Gstreamer programming. Gstreamer is a highly sophisticated software with a low-level API, specifically designed for building high-performance and reliable streaming applications.

To develop software using DeepStream, you would require expertise in machine learning, TensorRT, video processing, and Gstreamer. This can be a challenging task. However, Savant has been designed to address the last two points and provide a high-level, robust architecture built on top of Gstreamer/DeepStream. As a Savant user, you no longer need to be proficient in Gstreamer programming, as the framework's architecture solves most typical problems, simplifying the development process.

## Savant Architecture

Savant adds significant value to the DeepStream framework by introducing a higher level of abstraction built on top of DeepStream and Gstreamer. This layer is designed to simplify the development process by providing convenient abstractions, tools, and automation, which abstract away the underlying complexity of Gstreamer.

The picture illustrates the most important logical elements of Savant:

![Savant Architecture](https://user-images.githubusercontent.com/15047882/168019278-f75e5653-4332-4cd9-a54b-fa9e57902d26.png)

Some of the key components of the framework include:

**Framework Core**: The engine of the framework integrates with Gstreamer and DeepStream, enabling developers to create customized pipelines using YAML configuration and Python.

**Dynamic Configuration Manager**: This subsystem fetches parameters from external sources to configure pipeline elements during pipeline launch and execution.

**Libraries**: Savant provides custom preprocessing and postprocessing functions that are optimized for efficient computations within the framework.

**Virtual Streams Subsystem**: This subsystem is a critical component of the framework, representing external streams and auxiliary data in a unified format. It automatically configures necessary Gstreamer elements that handle dynamic data streams, optimizing performance, and collecting garbage when they are no longer needed.

**Pipeline Configurator**: This subsystem translates processing blocks specified with YAML to the Gstreamer graph that does the work. It includes standard model interfaces, custom preprocessing and postprocessing invocations, data selectors, and user-defined code in Python.

**Source Adapters**: Savant provides a set of data adapters that can inject frames from various media sources into the framework, increasing stability and reliability of the processing.

**Sink Adapters**: After being processed by the framework, the data is injected into an external system using a unified interface provided by the framework. With sink adapters, the framework converts and sends data into external systems safely, increasing stability and reliability.

In summary, Savant provides a comprehensive set of tools and subsystems that simplify the development of optimized video processing pipelines, making it an invaluable framework for developers working in this field.

## Pipeline Architecture

Every pipeline developed with Savant in the end looks like it is depicted on the following diagram:

![Savant Pipeline Architecture](https://user-images.githubusercontent.com/15047882/228788117-d8875670-ba7a-4468-b39e-6c699ae0d8dc.png)

Yellow blocks represent user-defined elements, while blue boxes represent everything framework provides the the developer.

## Virtual Streams Architecture

In Gstreamer and DeepStream, a stream is typically bound to a single data source, such as an RTSP camera. This stream is then integrated into the Gstreamer processing graph to perform specific processing tasks. For example, a simple Gstreamer pipeline may look like the one shown on the left side of the picture:

![image](https://user-images.githubusercontent.com/15047882/168025015-d49c4978-4b21-4170-ad2f-123a8d0bbc90.png)

In this pipeline, the file source represents the stream, and it is connected to a series of processing elements that perform specific tasks on the data. However, when dealing with more complex pipelines that involve multiple data sources and processing elements, it can become difficult to manage the pipeline efficiently.

Savant addresses this issue through its Virtual Streams Subsystem, which automatically represents any external media or sensor data in a unified format. This subsystem dynamically configures the necessary Gstreamer elements that handle dynamic data streams, optimizing performance and increasing stability. Additionally, it collects garbage when data is no longer needed, further optimizing the processing pipeline.

Savant's Virtual Streams Subsystem provides a high level of abstraction that simplifies the development of complex video processing pipelines, enabling developers to focus on their core tasks without the need for in-depth knowledge of Gstreamer programming.

When working with real-world dynamic sources, a naive approach to video processing can lead to multiple problems. One of the most significant drawbacks is the time it takes to load AI models into the GPU, making it impossible to utilize 100% of the GPU when reloading the pipeline after each file.

Another obstacle arises when dealing with unreliable data sources that may experience intermittent outages, such as an RTSP camera. To handle such events correctly, developers must create custom Gstreamer code that watches signals from the Gstreamer event bus, handles disconnect events, and dynamically reconstructs the pipeline to resume operation after an outage.

Additionally, when multiple data sources are processed in a multiplexed way within a single DeepStream/Gstreamer pipeline, a single source outage can cause the entire pipeline to crash, resulting in significant downtime and lost data.

Savant addresses these challenges by introducing a virtual stream abstraction. This abstraction decouples the Gstreamer sources from real-life sources, providing "always-on" virtual streams that are spawned by the framework, and data flow is switched to them. The framework then determines when these virtual streams are no longer in use and removes them, detaching them from the pipeline.

Real-life sources are never directly bound to the Gstreamer graph in Savant, so they only affect the corresponding adapters, while the pipeline is kept in memory and can process data from stable sources. This approach enables developers to handle dynamic sources and unreliable data without downtime or data loss, improving the overall efficiency and reliability of the video processing pipeline.

![Savant Virtual Sources](https://user-images.githubusercontent.com/15047882/168033994-0da8304f-cb02-4fd0-a9c2-5c8636367a4e.png)

Muxed Virtual Stream block accepts the streaming data from the outside world via [ZeroMQ](https://zeromq.org/) socket. The framework supports three kinds of sockets:
- Pub/Sub - when one needs to drop the excessive input data but run the inference in real-time (fits for processing of already-decoded, raw streams);
- Dealer/Router - when one needs to utilize back-pressure to stop the source from injecting the data too fast but don't care about real-time processing;
- Req/Rep - when one needs to ensure that every frame passed is loaded into the pipeline before the following frame from the same source can be passed.

Virtual Stream Manager investigates the packets from the Muxed Virtual Stream and dynamically manages corresponding Gstreamer elements to integrate streams into the pipeline.

Each message received by Muxed Virtual Stream is in Savant-RS format, including stream information, payload, and metadata attributes. While stream information and payload are mandatory, metadata attributes are optional elements that provide two features:
- ability to adjust the processing on a frame-by-frame basis;
- ability to split pipelines into a decoupled parts;

As an example of per-frame metadata, you could imagine a temperature sensor or lightness sensor information that tunes the AI algorithm to behave correctly.

As an example of decoupling, you can imagine a pipeline that requires enriching the data in the middle. The first part of the pipeline produces intermediate results, passing them into the free-form software, injecting additional metadata attributes, filtering them, and sending the data to the second part of the pipeline. Such design helps to utilize GPU resources at most; otherwise, if to try implementing such a pipeline in an all-in-one way, you face the IO latencies which affect GPU utilization efficiency. The rule of thumb is to avoid external interactions from the pipeline.

![metadata](https://user-images.githubusercontent.com/15047882/168040020-e87f288d-8cad-4b6c-8ec7-a27dc4b649f5.png)

Sink streams are also muxed and virtualized. The data is in Savant-RS format and is sent to a single ZeroMQ socket, which also can be:
- Pub/Sub - when one needs to drop the excessive output data but continue running the inference in real-time (fits for processing of already-decoded, raw streams);
- Dealer/Router - when one needs to use back-pressure to stop the pipeline from sending the data too fast because receiving part is too slow to keep up;
- Req/Rep - when one needs to ensure that the packet is delivered to the receiver before the next one is transferred.

## Source Adapters

We developed several data source adapters that fit daily development and production use cases. The developers can also use them as a source base for creating custom adapters - the apparent scenario is implementing an adapter that mixes external data into frames providing additional data to the AI pipeline. 

Every adapter is designed for use with a particular data source: 

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

Since the adapter module is decoupled from the pipeline, its launch is not expensive. Adapter crash also doesn't affect the pipeline execution. Local and remote adapters for video files support both sync and as-fast-as-possible models of data injection. 

The first mode sends data in the pipeline with the FPS specified in the file - it's convenient when testing real-time execution GPU utilization or visual inspection. The second mode is used when the purpose is to process as much data as possible utilizing all available GPU resources.

## Sink Adapters

Sink adapters send inference data and(or) generated multimedia data into external storage for future use. Unfortunately, there are many such storages, and schemes of stored data may vary even within a single system. That's why it's impossible to build universal sink adapters that fit all needs. So instead, we developed several sinks which can be easily modified to create the one you need in your system:

A developer would be especially interested in the following sinks:

- [Inference results are placed into JSON file stream](docs/adapters.md#the-json-meta-sink-adapter);
- [Resulting video overlay displayed on a screen (per source)](docs/adapters.md#the-display-sink-adapter);
- [MP4 file (per source)](docs/adapters.md#the-video-file-sink-adapter);
- [image directory (per source)](docs/adapters.md#the-image-file-sink-adapter);
- [Always-On RTSP Stream Sink](docs/adapters.md#the-always-on-rtsp-sink-adapter).

## Pipeline Configurator

The pipeline configurator translates the YAML pipeline into a proper Gstreamer graph with DeepStream plugins. 

The pipeline is defined in the module.yml manifest file and includes the following blocks:
- a module name;
- global parameters block;
- a pipeline block consisting of a source, a sink, and processing elements.

There are number of processing element types that can form a pipeline:
- detector;
- attribute_model;
- pyfunc;
- complex_model;
- ... etc ...

Every element represents a graph node that handles data passing through it. The element of a type (like detector) has parameters that configure its behavior. The parameters for every element type are described in the documentation.

Every element does the following operations on data:
- selects/filters the incoming data units for processing;
- processes selected data;
- transforms metadata with new attributes.

The following image displays how it works:

![processing](https://user-images.githubusercontent.com/15047882/168789581-d57fa66b-b4c6-4226-9bfa-fcfc9bfe95ca.png)

Let's briefly walk through the data processing. In step (1), you can see the unit of data that moves through the pipeline. The data unit consists of a media frame in GPU RAM (typically for video) and metadata allocated in CPU RAM. 

When the data unit reaches the element of a pipeline, element selector (2) filters whether the element must process the data unit or not. Remember, the selector doesn't **drop** units from the further processing, not selected units just pass through the element. 

_E.g. The previous element was YOLO, and the selector of the current element selects only person class with confidence 0.5 and higher to find hardhats._

If so, the unit media frame and required metadata are passed to the processing block of the element (3) - you can see the inference block in the diagram; however it can be another kind of processing, e.g., classic CV algorithm run in CPU or GPU.

Processing element (3) generates certain output like tensors or bounding boxes, attributes that are passed back to postprocessing (4). 

The postprocessing block may drop, mangle or modify metadata to match the needs of the subsequent elements (5).

_E.g. The hard hat model produces hard hat bounding boxes with confidence. The postprocessing block only outputs those with confidence greater than 0.6._

The framework joins existing metadata records passed through the element (6) with those produced by the element (5).

Figure (7) shows that after the element is applied to the data unit still has a media frame located in GPU RAM and metadata records in CPU RAM. The frame can be the same if the element does not change the frame data or if the element changes the frame, it is changed.

The whole pipeline of elements works as shown in the picture:

![module](https://user-images.githubusercontent.com/15047882/168824194-fa7da94a-6a99-443d-814f-4d16b38d8aee.png)

Ideally, heavyweight image frames do not leave the GPU RAM, being handled by CUDA kernels, while the system CPU executes metadata operations. However, sometimes, the image frames or at least parts of them may be copied to the CPU. When an element doesn't have a CUDA kernel implemented, less-efficient processing is accomplished in the system CPU. Another case is when a highly efficient model implemented in CPU exists - e.g., a model from OpenVino Zoo, so it can be effective to copy part of a frame to run CPU inference. 
