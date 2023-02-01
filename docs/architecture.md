# Architecture Overview

Savant is built on top of DeepStream SDK - state-of-the-art Nvidia framework for streaming AI applications. DeepStream is the core framework of the Nvidia ecosystem because it unleashes the power of Nvidia accelerators in inference tasks. 

No other general-purpose framework is able to reach performance comparable to DeepStream in inference tasks for video processing. There is a reason for that: popular frameworks like PyTorch, TensorFlow, OpenCV, and FFmpeg are bound to the CPU when doing the operations, while DeepStream uses the CPU only to control the flow. It's not obvious, but moving image frames between CPU and GPU takes a lot of time and decreases the performance dramatically.

In the DeepStream, operations like video decoding, frame transformations, inference, and finally encoding (if necessary) run in GPU memory and use GPU hardware features to speed up the process reaching the limits of the accelerator card. Building such kind of a framework is a very sophisticated task that requires a broad knowledge of Nvidia hardware and low-level tools like CUDA and TensorRT.

![Nvidia DeepStream Architecture](https://user-images.githubusercontent.com/15047882/167308102-eea0915d-e1e5-4924-bd4c-69da34d47fc7.png)

Unfortunately, DeepStream has a very steep learning curve which is a showstopper for most ML engineers because it introduces a very advanced knowledge domain - Gstreamer programming. Gstreamer is very sophisticated software with a pretty low-level API that is developed to build high-performance, reliable streaming applications.

So, to build your software on top of DeepStream, you have to have:
1. machine-learning expertise (probably you have);
2. TensorRT expertise;
3. Video processing expertise;
4. Gstreamer expertise;

That sounds tricky. Savant is designed to address the last two points with a very high-level and robust architecture built on top of Gstreamer/DeepStream. As a framework user, you no longer care about Gstreamer programming because the framework solves most typical problems in its architecture.

## Savant Architecture

Savant introduces valuable extensions to the DeepStream framework through the abstraction level built above DeepStream and Gstreamer. The purpose of that layer is to hide all the complexity of Gstreamer with handy abstractions, tools, and automation. One can see the most important logical elements in the picture:

![Savant Architecture](https://user-images.githubusercontent.com/15047882/168019278-f75e5653-4332-4cd9-a54b-fa9e57902d26.png)

**Framework Core**. The framework's engine integrates with Gstreamer and DeepStream, allowing developers to create extensions and programmatic implementations of customized pipelines without use of Savant's declarative YAML architecture.

**Dynamic Configuration Manager**. The subsystem fetches parameters from the outer space to configure pipeline elements during the pipeline launch and execution.

**Libraries**. We have developed custom preprocessing and postprocessing functions for efficient computations delivered with the framework.

**Virtual Streams Subsystem**. One of the most valuable parts of the framework which doing magic. We will cover it in depth later. But, to get an idea, the subsystem represents any external media or sensor data in a unified format; automatically configures necessary Gstreamer elements that handle dynamic data streams, and collects the garbage when they are no longer needed.

**Pipeline Configurator**. The subsystem translates processing blocks specified with YAML to the Gstreamer graph that does the work. Those blocks include standard model interfaces, custom preprocessing and postprocessing invocations, data selectors, and user-defined code in Python.

**Source Adapters**. Set of data adapters that can inject frames from various media sources into the framework. Source adapters are also excellent examples of how to build your adapter. Decoupling source adapters from the framework increases the stability and reliability of the processing.

**Sink Adapters**. After being processed by the framework, the data is injected into an external system. The framework provides the unified interface for that. With sink adapters framework converts and sends the data into external systems safely. Decoupling sink adapters from the framework also increases the stability and reliability of the processing.

## Virtual Streams Architecture

The Gstreamer and DeepStream stream is bound to a single data source. E.g., a stream can represent an RTSP camera. Therefore, that stream is a part of the Gstreamer processing graph. For example, when dealing with a simple Gstreamer pipeline, it looks like shown on the left side of the picture:

![image](https://user-images.githubusercontent.com/15047882/168025015-d49c4978-4b21-4170-ad2f-123a8d0bbc90.png)

It works well for static pipelines, which process "reliable" data sources like files located in a filesystem. In the scenario, the pipeline is launched for a particular file, processes it, and finally exits at the end of the processing.

However, the above example is very primitive and has many drawbacks. Let's look at them. 

The first obvious drawback is that loading AI models in GPU takes some time, and thus it's impossible to utilize 100% of GPU when reloading the pipeline after each file.

The second one happens when you use an unreliable data source that flaps from time to time (e.g., RTSP camera). To handle it correctly, you have to create custom Gstreamer code that watches signals from the Gstreamer event bus, handles such disconnect events, dynamically reconstruct the pipeline, and resume the operation after.

Without coding such processing, Gstreamer will crash even if it's OK to restart; it takes some time to load AI models in GPU RAM.

Finally, we face the real problem when multiple data sources are processed in a multiplexed way within a single DeepStream/Gstreamer pipeline. Practically it's a commonplace scenario: the DeepStream is blazing fast, so it can process a lot of data. Users inject data from multiple sources into a single pipeline to load it well. That leads to a growth of outages because if a single source is out of order, the whole pipeline crashes without proper signal processing.

Savant handles all such cases by providing a virtual stream abstraction. That abstraction introduces Gstreamer sources that are "always-on" because they are decoupled from real-life sources. The framework spawns those virtual streams, switches data flow to them, determines when they are no longer in use, and wipes them, detaching them from the pipeline.

In Savant, real-life sources are never directly bound to the Gstreamer graph, so they affect only corresponding adapters. Meanwhile, the pipeline is kept in memory and can process data from stable sources.

![Savant Virtual Sources](https://user-images.githubusercontent.com/15047882/168033994-0da8304f-cb02-4fd0-a9c2-5c8636367a4e.png)

Muxed Virtual Stream block accepts the streaming data from the outside world via [ZeroMQ](https://zeromq.org/) socket. The framework supports two kinds of sockets:
- Pub/Sub - when you would like to drop the excessive input data but run the inference in real-time;
- Push/Pull - when you would like to use back-pressure to stop the source from injecting the data too fast but don't care about real-time processing.

Virtual Stream Manager investigates the packets from the Muxed Virtual Stream and dynamically manages corresponding Gstreamer elements to integrate streams into the pipeline.

Each message received by Muxed Virtual Stream is in AVRO format, including stream information, payload, and metadata attributes. While stream information and payload are mandatory, metadata attributes are optional elements that provide two features:
- ability to adjust the processing on a frame-by-frame basis;
- ability to split pipelines into a decoupled parts;

As an example of per-frame metadata, you could imagine a temperature sensor or lightness sensor information that tunes the AI algorithm to behave correctly.

As an example of decoupling, you can imagine a pipeline that requires enriching the data in the middle. The first part of the pipeline produces intermediate results, passing them into the free-form software, injecting additional metadata attributes, filtering them, and sending the data to the second part of the pipeline. Such design helps to utilize GPU resources at most; otherwise, if to try implementing such a pipeline in an all-in-one way, you face the IO latencies which affect GPU utilization efficiency. The rule of thumb is to avoid external interactions from the pipeline.

![metadata](https://user-images.githubusercontent.com/15047882/168040020-e87f288d-8cad-4b6c-8ec7-a27dc4b649f5.png)

Sink streams are also muxed and virtualized. The data is in AVRO format and is sent to a single ZeroMQ socket, which also can be:
- Pub/Sub - when you would like to drop the excessive output data but continue running the inference in real-time;
- Push/Pull - when you would like to use back-pressure to stop the pipeline from sending the data too fast because receiving part is too slow to keep up.

## Source Adapters

We developed several data source adapters that fit daily development and production use cases. The developers can also use them as a source base for creating custom adapters - the apparent scenario is implementing an adapter that mixes external data into frames providing additional data to the AI pipeline. 

Every adapter is designed for use with a particular data source: 
- local or remote video file; 
- directory of video files, 
- local or remote image file;
- directory of images;
- RTSP stream;
- USB/CSI camera stream;
- Image/Video Stream in Apache Kafka.

Since the adapter is decoupled from the pipeline, its launch is not expensive. Adapter crash also doesn't affect the pipeline execution. Local and remote adapters for video files support both sync and as-fast-as-possible models of data injection. 

The first mode sends data in the pipeline with the FPS specified in the file - it's convenient when testing real-time execution GPU utilization or visual inspection. The second mode is used when the purpose is to process as much data as possible utilizing all available GPU resources.

## Sink Adapters

Sink adapters send inference data and(or) generated multi-media data into external storage for future use. Unfortunately, there are many such storages, and schemes of stored data may vary even within a single system. That's why it's impossible to build universal sink adapters that fit all needs. So instead, we developed several sinks which can be easily modified to create the one you need in your system:

A developer would be especially interested in the following sinks:
- JSON metadata sink saves inference results into JSON records in a file;
- video file sink saves the resulting video or image set into a video file;
- video play sink displays video or pictures on screen;

Following sinks can be used in production as-is:
- Elasticsearch sink writes resulting inference data into Elasticsearch index;
- MongoDB sink writes resulting inference data into MongoDB document index;
- Kafka sink writes resulting inference data into metadata Kafka topic and resulting video into video Kafka topic.

## Pipeline Configurator

The pipeline configurator translates the YAML pipeline into a proper Gstreamer graph with DeepStream plugins. 

The pipeline is defined in the module.yml manifest file and includes the following blocks:
- a module name;
- global parameters block;
- a pipeline block consisting of a source, a sink, and processing elements.

There are number of processing element types that can form a pipeline:
- detector;
- attribute_model;
- rotated_object_detector;
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

## Dynamic Pipeline Configuration

TODO
