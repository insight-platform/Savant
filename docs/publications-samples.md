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

A simple pipeline that detects the people within the camera's viewport and counts those who crossed the line in both directions. The sample utilizes Nvidia PeopleNet Model and the Nvtracker DeepStream plugin to track the persons.

Please find the code and instructions in the [Repository](#).
In addition, a step-by-step guide published on the Medium Portal is also [available](#).

Tested Platforms are:
- Jetson Nano, Xavier NX, Xavier AGX;
- Nvidia Turing;
- Nvidia Ampere.

The work results are demonstrated on multiple fake RTSP streams.

### Hard-Hat People Counter With OCR

### Camera Scene Change Detector

### Sophisticated Person Profiler and Tracker

