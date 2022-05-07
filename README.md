# Savant

Savant is an open-source, high-level toolkit for implementing real-time, streaming, highly efficient multimedia AI applications on the Nvidia stack. It makes it possible to develop the inference pipelines which utilize the best Nvidia approaches for both data center and edge accelerators with a very moderate amount of code in Python.

[Nvidia DeepStream](https://developer.nvidia.com/deepstream-sdk) is today's most advanced toolkit for developing high-performance real-time streaming AI applications that run several times faster than conventional AI applications executed within the conventional runtimes like PyTorch, TensorFlow and similar. 

That is possible due to specially designed architecture, which utilizes the best Nvidia accelerator features, including hardware encoding and decoding for video streams, moving the frames thru inference blocks mostly in GPU RAM without excessive data transfers between CPU and GPU RAM. It also stands on a highly optimized low-level software stack that optimizes inference operations to get the best of the hardware used.

## Why?

Why choose Savant if DeepStream solves the problem? Because DeepStream is a very tough and challenging to use technology.

The root of that is that DeepStream is implemented as a set of plug-ins for Gstreamer - the open-source multi-media framework for highly-efficient streaming applications. It makes developing more or less sophisticated DeepStream applications very difficult because the developer must understand how the Gstreamer processes the data. That makes the learning curve very steep.

Savant is a very high-level framework over the DeepStream, which incapsulates all the Gstreamer internals from the developer and provides a beneficial tools for implementing real-life streaming AI inference applications. So, as a developer, you implement your inference pipeline as a set of declarative (YAML) blocks with several user-defined functions in Python (and C/C++ if you would like to utilize the most of CUDA runtime).

The Savant framework provides many really needed features out of the box. When we created Savant, we utilized the best of our experience, which we achieved through several years of commercial DeepStream development.

### Dynamic Pipeline Configuration

The framework gives the developer the means to configure and reconfigure the inference pipeline operation dynamically with:
- incoming per-frame metadata;
- Etcd variables which a watched and instantly applied;
- 3rd-party sources that are accessible through user-defined functions.

