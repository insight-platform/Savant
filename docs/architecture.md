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

![Savant Architecture](https://user-images.githubusercontent.com/15047882/168017486-a8cd71a1-2154-426e-990c-eec27ac72937.png)


**Framework Core**. The framework's engine integrates with Gstreamer and DeepStream, allowing developers to create extensions and programmatic implementations of customized pipelines outside Savant's declarative architecture.

**Dynamic Configuration Manager**. The subsystem fetches parameters from the outer space to configure pipeline elements during the pipeline launch and execution.

**Libraries**. We have developed custom preprocessing and postprocessing functions for efficient computations delivered with the framework.

**Virtual Streams Subsystem**. One of the most valuable parts of the framework which doing magic. We will cover it in depth later. But, to get an idea, the subsystem represents any external media or sensor data in a unified format, automatically rolling out necessary processing elements of Gstreamer that process the data elements and running garbage collection when they are no longer needed.  
