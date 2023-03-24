# Savant Demos, Samples and Education Materials

On this page, you can find practical examples, how-tos, publications, and other non-formal documentation links that can
help you dive into Savant.

- [ML/AI Examples](#mlai-examples)
- [Utility And Coding Examples](#utility-and-coding-examples)

## ML/AI Examples

To help novice users to dive into Savant, we have prepared several ready-to-use examples, which one can download, build
and launch. In addition, every sample includes all you need to run it: the source code, models, build file, sample data,
and a short README.md which covers the purpose and gives a brief explanation. Some samples are also accompanied by the
how-to guides published on the Medium portal, where one can study them step-by-step.

### People Detecting, Tracking and Anonymizing (Peoplenet, Nvidia Tracker, OpenCV-CUDA)

![demo-0 2 0 (2)](https://user-images.githubusercontent.com/15047882/227564451-a43fccec-f9bb-49b0-b11b-9d9e67177e92.png)

A simple pipeline that uses standard [Nvidia Peoplenet]([https://github.com/pjreddie/darknet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)) to detect persons and their faces in the video. Then the faces are matched versus bodies and blurred with the integrated OpenCV CUDA functionality. There is also a simple not reliable tracker that helps reduce flickering of boxes.

The **Green** Icon represents how many people with blurred faces in the scene. 
The **Blue** Icon represents how many people with blurred faces in the scene.

YouTube Video:

[![Watch the video](https://img.youtube.com/vi/YCvT3XbiSik/default.jpg)](https://youtu.be/YCvT3XbiSik)

A step-by-step [tutorial](#).

Code and simple instructions in the [Demo Directory](../samples/peoplenet_detector).

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Ampere.

Demonstrated operational modes:

- real-time processing: RTSP streams (multiple sources at once);
- capacity processing: directory of files (FPS).

Demonstrated adapters:
- RTSP source adapter;
- video file source adapter;
- Always-ON RTSP sink adapter;
- Video/Metadata sink adapter.
