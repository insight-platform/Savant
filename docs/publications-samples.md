# Publications And Samples

On this page, you can find practical examples, how-tos, publications, and other non-formal documentation links that can
help you dive into Savant.

## Examples

To help novice users to dive into Savant, we have prepared several ready-to-use examples, which one can download, build
and launch. In addition, every sample includes all you need to run it: the source code, models, build file, sample data,
and a short README.md which covers the purpose and gives a brief explanation. Some samples are also accompanied by the
how-to guides published on the Medium portal, where one can study them step-by-step.

### Object Detector (YOLOv4)

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

Pretty Sophisticated example which detects the person with PeopleNet, next detects the hard hat within the person 
bounding box with specially trained RetinaNet model, assigns hard hats to persons, and finally runs EasyOCR PyTorch 
code to recognize the mark on a hard hat. The persons are tracked on the scene with DeepStream's Nvtracker plugin. 

The line crossing count is done either with Nvdsanalytics plugin which gives false results pretty often or with 
custom UDF function which gives results of a better quality.

Please find the code and instructions in the [Repository](#).
In addition, a step-by-step guide published on the Medium Portal is also [available](#).

Tested Platforms are:

- Jetson Nano, Xavier NX, Xavier AGX;
- Nvidia Turing;
- Nvidia Ampere.

The results are demonstrated on multiple fake RTSP streams.

### Camera Scene Change Detector

### Sophisticated Person Profiler and Tracker

