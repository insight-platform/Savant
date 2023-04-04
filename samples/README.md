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

Sample Location: [peoplenet_detector](./peoplenet_detector)

Preview:

![](peoplenet-blur-demo-loop.webp)

Features:

![demo-0 2 0-dia](https://user-images.githubusercontent.com/15047882/229354155-f37ad24b-c0bd-4249-ba83-a53b3c1e053b.png)

A simple pipeline that uses standard [Nvidia PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) model to detect persons and their faces in the video. The faces are matched versus bodies and blurred with the integrated OpenCV CUDA functionality. There is also a simple unreliable tracker that helps reduce flickering of boxes.

GPU-accelerated blurring made with OpenCV is a killer feature of this demonstration. It enables very fast and efficient face blurring without CPU utilization.

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

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/peoplenet_detector

# if you want to share with us where are you from
# run the following command, it is completely optional
curl --silent -O -- https://hello.savant.video/peoplenet.html

# if x86
../../utils/check-environment-compatible && \
   docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && \
   docker compose -f docker-compose.l4t.yml up

# Ctrl+C to stop the running compose bundle

# to get back to project root
cd ../..
```
