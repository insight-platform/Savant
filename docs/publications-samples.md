# Publications And Samples

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

![features](https://user-images.githubusercontent.com/15047882/227521582-49ce7c8b-3b67-4524-a298-da8aab7110ef.png)

A simple pipeline that uses standard [Nvidia Peoplenet]([https://github.com/pjreddie/darknet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)) to detect persons and their faces in the video. Then the faces are matched versus bodies and blurred with the integrated OpenCV CUDA functionality. There is also a simple not reliable tracker that helps reduce flickering of boxes.


Please find the code and instructions in the [Demo Directory](../samples/peoplenet_detector).
In addition, a step-by-step guide published on the Medium Portal is also [available](#).

Tested Platforms are:

- Xavier NX, Xavier AGX;
- Nvidia Ampere.

Operational Modes are:

- real-time processing: fake RTSP stream;
- capacity processing: directory of files.

