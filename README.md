# Savant

Savant is an open-source, high-level toolkit for implementing real-time, streaming, highly efficient multimedia AI applications on the Nvidia stack. It makes it possible to develop the inference pipelines which utilize the best Nvidia approaches for both data center and edge accelerators with a very moderate amount of code in Python.

[Nvidia DeepStream](https://developer.nvidia.com/deepstream-sdk) is today's most advanced toolkit for developing high-performance real-time streaming AI applications that run several times faster than conventional AI applications executed within the conventional runtimes like PyTorch, TensorFlow and similar. 

That is possible due to specially designed architecture, which utilizes the best Nvidia accelerator features, including hardware encoding and decoding for video streams, moving the frames thru inference blocks mostly in GPU RAM without excessive data transfers between CPU and GPU RAM. It also stands on a highly optimized low-level software stack that optimizes inference operations to get the best of the hardware used.

## Why?

Why choose Savant if DeepStream solves the problem? Because DeepStream is a very tough and challenging to use technology.

The root of that is that DeepStream is implemented as a set of plug-ins for Gstreamer - the open-source multi-media framework for highly-efficient streaming applications. It makes developing more or less sophisticated DeepStream applications very difficult because the developer must understand how the Gstreamer processes the data. That makes the learning curve very steep.

Savant is a very high-level framework over the DeepStream, which incapsulates all the Gstreamer internals from the developer and provides a beneficial tools for implementing real-life streaming AI inference applications. So, as a developer, you implement your inference pipeline as a set of declarative (YAML) blocks with several user-defined functions in Python (and C/C++ if you would like to utilize the most of CUDA runtime).

The Savant framework provides many really needed features out of the box. When we created Savant, we utilized the best of our experience, which we achieved through several years of commercial DeepStream development.

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

## Rotated Bounding Boxes Out Of The Box

In our practice, when we create commercial AI software, we often meet the cases where the detection bounding boxes are rotated relative to a video frame. It is often the case when the camera observes the viewport from the ceiling when the objects are situated on the floor. 

These cases require detecting the objects in the way the parts of other things hit the bounding box minimally. To achieve that, unique models which calculate box angle are used [RAPiD](https://vip.bu.edu/projects/vsns/cossy/fisheye/rapid/).

![image1](https://vip.bu.edu/files/2020/05/Edge_teaser_w_count.gif)
![image2](https://vip.bu.edu/files/2020/05/RAPiD_1024_Crowd_exhibition_w_count.gif)

Such models require additional post-processing, which involves the rotation because otherwise, you cannot utilize most of the classifier models as they need orthogonal boxes as their input.

Savant supports rotated bounding boxes out of the box as well as the means to handle rotated bounding boxes right before they are passed to the classifier models.
