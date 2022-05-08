# Architecture

Savant is built on top of DeepStream SDK - state of an art Nvidia framework for streaming AI applications. DeepStream is the core framework of the Nvidia ecosystem because it allows unleashing the power of Nvidia accelerators in inference tasks. 

No other general-purpose framework is able to reach comparable to DeepStream performance in inference tasks for video processing. There is a reason for that: popular frameworks like PyTorch, TensorFlow, OpenCV, and FFmpeg are bound to the CPU when doing the operations, while DeepStream uses the CPU only to control the flow. It's not obvious, but moving images between CPU and GPU takes a lot of time and decreases the performance dramatically.

In the DeepStream, operations like video decoding, frame transformations, inference, and finally encoding (if necessary) run in GPU memory and use GPU hardware features to speed up the process to reach the limits of the accelerator card. Building such kind of a framework is a very difficult task that requires a broad knowledge of Nvidia hardware and low-level tools like CUDA and TensorRT.

![Nvidia DeepStream Architecture](https://user-images.githubusercontent.com/15047882/167308102-eea0915d-e1e5-4924-bd4c-69da34d47fc7.png)

Unfortunately, DeepStream is still very sophisticated for a regular AI/ML engineer because it introduces a very sophisticated knowledge domain - Gstreamer programming. Gstreamer is very advanced software with a pretty complex API that is developed to build reliable streaming applications.

So, to build your software on top of DeepStream, you have to have:
1. machine-learning expertise (probably you have);
2. TensorRT expertise;
3. Video processing expertise;
4. Gstreamer expertise;

Sounds sophisticated. Savant tries to address the last two points with a very simple and robust architecture.
