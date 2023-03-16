# Publications And Samples

THIS IS WIP SECTION. DON'T BELIEVE 100% TO WHAT YOU SEE HERE.

On this page, you can find practical examples, how-tos, publications, and other non-formal documentation links that can
help you dive into Savant.

- [ML/AI Examples](#mlai-examples)
- [Utility And Coding Examples](#utility-and-coding-examples)

## ML/AI Examples

To help novice users to dive into Savant, we have prepared several ready-to-use examples, which one can download, build
and launch. In addition, every sample includes all you need to run it: the source code, models, build file, sample data,
and a short README.md which covers the purpose and gives a brief explanation. Some samples are also accompanied by the
how-to guides published on the Medium portal, where one can study them step-by-step.

### Object Detector (YOLOv4)

![image](https://user-images.githubusercontent.com/15047882/167287037-afeab5a3-8d61-477e-b6c6-9128636026b8.png)

A minimalistic pipeline that uses standard [YOLOv4](https://github.com/pjreddie/darknet) to detect objects in the video.
It's an excellent place to begin diving into Savant.

Please find the code and instructions in the [Repository](#).
In addition, a step-by-step guide published on the Medium Portal is also [available](#).

Tested Platforms are:

- Jetson Nano, Xavier NX, Xavier AGX;
- Nvidia Turing;
- Nvidia Ampere.

Operational Modes are:

- real-time processing: fake RTSP stream, MJPEG/RGBA USB-camera;
- capacity processing: directory of files.

### People Counter (PeopleNet + Nvidia Tracker)

![image](https://user-images.githubusercontent.com/15047882/167286974-b620d64c-cee3-4922-8809-28454284c916.png)

A simple pipeline that detects the people within the camera's viewport and counts those who crossed the line in both
directions. The sample utilizes Nvidia PeopleNet Model and the Nvtracker DeepStream plugin to track the persons.

Please find the code and instructions in the [Repository](#).
In addition, a step-by-step guide published on the Medium Portal is also [available](#).

Tested Platforms are:

- Jetson Nano, Xavier NX, Xavier AGX;
- Nvidia Turing;
- Nvidia Ampere.

The results are demonstrated on multiple fake RTSP streams.

### FishEye Camera People Detector and Tracker (RAPiD + SORT)

[![image](https://user-images.githubusercontent.com/15047882/167245173-aa0a18cd-06c9-4517-8817-253d120c0e07.png)](#)

The sample demonstrates how detector models that result in rotated bounding boxes work in Savant. The detector model
used in the example is [RAPiD](https://vip.bu.edu/projects/vsns/cossy/fisheye/rapid/). The tracker is a modified SORT
implementation that can track such bounding boxes on the scene.

Please find the code and instructions in the [Repository](#).
In addition, a step-by-step guide published on the Medium Portal is also [available](#).

Tested Platforms are:

- Jetson Nano, Xavier NX, Xavier AGX;
- Nvidia Turing;
- Nvidia Ampere.

The results are demonstrated on multiple fake RTSP streams.

### Hard-Hat People Counter With OCR

![image](https://user-images.githubusercontent.com/15047882/167287122-9709230e-e3d8-4740-9b0a-1a47419fcd30.png)

A pretty sophisticated example that looks for the persons with PeopleNet, next it detects the hard hat within the person
bounding box with custom trained RetinaNet model, assigns hard hats to persons, and finally runs EasyOCR PyTorch code to
recognize the mark on a hard hat. Finally, the persons are tracked on the scene with DeepStream's Nvtracker plugin.

The line crossing count is done either with the Nvdsanalytics plugin, which gives false results or with a custom UDF
function, which provides better quality results.

Please find the code and instructions in the [Repository](#).
In addition, a step-by-step guide published on the Medium Portal is also [available](#).

Tested Platforms are:

- Jetson Nano, Xavier NX, Xavier AGX;
- Nvidia Turing;
- Nvidia Ampere.

The results are demonstrated on multiple fake RTSP streams.

### Camera Scene Change Detector

WIP

### Car Plate Detector (Nvdia)

WIP

## Utility And Coding Examples

This section includes examples that solve particular problems met in practical tasks. To understand them, we advise you
to get familiar with [ML/AI Examples](#mlai-examples) to feel free to implement the pipelines with Savant.

### Etcd Dynamic Pipeline Configuration

WIP

### Implementing Custom Source & Sink Adapters

WIP

### Kafka Source-Sink Adapter

WIP

### Bounding Box Rotation With Savant

WIP
