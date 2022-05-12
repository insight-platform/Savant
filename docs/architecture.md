# Architecture

Savant is built on top of DeepStream SDK - state of an art Nvidia framework for streaming AI applications. DeepStream is the core framework of the Nvidia ecosystem because it allows unleashing the power of Nvidia accelerators in inference tasks. 

No other general-purpose framework is able to reach comparable to DeepStream performance in inference tasks for video processing. There is a reason for that: popular frameworks like PyTorch, TensorFlow, OpenCV, and FFmpeg are bound to the CPU when doing the operations, while DeepStream uses the CPU only to control the flow. It's not obvious, but moving images between CPU and GPU takes a lot of time and decreases the performance dramatically.

In the DeepStream, operations like video decoding, frame transformations, inference, and finally encoding (if necessary) run in GPU memory and use GPU hardware features to speed up the process to reach the limits of the accelerator card. Building such kind of a framework is a very difficult task that requires a broad knowledge of Nvidia hardware and low-level tools like CUDA and TensorRT.

![Nvidia DeepStream Architecture](https://user-images.githubusercontent.com/15047882/167308102-eea0915d-e1e5-4924-bd4c-69da34d47fc7.png)

Unfortunately, DeepStream is still very steep for an ML engineer because it introduces a very sophisticated knowledge domain - Gstreamer programming. Gstreamer is very advanced software with a pretty complex API that is developed to build reliable streaming applications.

So, to build your software on top of DeepStream, you have to have:
1. machine-learning expertise (probably you have);
2. TensorRT expertise;
3. Video processing expertise;
4. Gstreamer expertise;

Sounds sophisticated. Savant is designed to address the last two points with a very simple and robust architecture built on top of Gstreamer/DeepStream - as a framework user you no longer care about Gstreamer programming, because the framework solves most typical problems in its architecture.

## Savant Architecture

Savant introduces valuable extensions to the DeepStream framework through the abstraction level built above DeepStream and Gstreamer. The purpose of that layer is to hide all the complexity of Gstreamer with handy abstractions, tools, and automation. One can see the most important logical elements in the picture:

![Savant Architecture](https://user-images.githubusercontent.com/15047882/168019278-f75e5653-4332-4cd9-a54b-fa9e57902d26.png)

**Framework Core**. The framework's engine integrates with Gstreamer and DeepStream, allowing developers to create extensions and programmatic implementations of customized pipelines outside Savant's declarative architecture.

**Dynamic Configuration Manager**. The subsystem fetches parameters from the outer space to configure pipeline elements during the pipeline launch and execution.

**Libraries**. We have developed custom preprocessing and postprocessing functions for efficient computations delivered with the framework.

**Virtual Streams Subsystem**. One of the most valuable parts of the framework which doing magic. We will cover it in depth later. But, to get an idea, the subsystem represents any external media or sensor data in a unified format, automatically rolling out necessary processing elements of Gstreamer that process the data elements and running garbage collection when they are no longer needed.  

**Pipeline Configurator**. The subsystem translates processing blocks specified with YAML to the Gstreamer graph that does the work. Those blocks include standard model interfaces, custom preprocessing and postprocessing invocations, data selectors, and user-defined code in Python.

**Source Adapters**. Set of data adapters developed to fit framework architecture that can inject frames from various media sources into the framework. Source adapters are also excellent examples of how to build your adapter. Decoupling source adapters from the framework also increases the stability and reliability of the processing.

**Sink Adapters**. After being processed by the framework, the data is injected into an external system. The framework provides the unified interface for that. With adapters framework converts and sends the data into external systems safely. Decoupling sink adapters from the framework also increases the stability and reliability of the processing.

## Virtual Stream Architecture

todo
