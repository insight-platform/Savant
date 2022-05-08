# Architecture

Savant is built on top of DeepStream SDK - state of an art Nvidia framework for streaming AI applications. DeepStream is the core framework of the Nvidia ecosystem because it allows unleashing the power of Nvidia accelerators in inference tasks. 

No other general-purpose framework is able to reach comparable to DeepStream performance in inference tasks for video processing. There is a reason for that: popular frameworks like PyTorch, TensorFlow, OpenCV, and FFmpeg are bound to the CPU when doing the operations, while DeepStream uses the CPU only to control the flow. It's not obvious, but moving images between CPU and GPU takes a lot of time and decreases the performance dramatically.

In the DeepStream, operations like video decoding, frame transformations, inference, and finally encoding (if necessary) run in GPU memory and use GPU hardware features to speed up the process to reach the limits of the accelerator card. Building such kind of a framework is a very difficult task that requires a broad knowledge of Nvidia hardware and low-level tools like CUDA and TensorRT.
