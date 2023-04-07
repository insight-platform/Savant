# People detection, tracking and face blurring (PeopleNet, Nvidia Tracker, OpenCV CUDA)

A simple pipeline that uses MOG2 from OpenCV to remove background from the frame. 
GPU-accelerated of background removal made with OpenCV cuda. 
It enables very fast and efficient remove background.

The left part represents the reference video. 
The right part represents video without background.

Preview:

![](../peoplenet-blur-demo-loop.webp)

Code and simple instructions in the [Demo Directory](../samples/background_remover).

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated operational modes:

- real-time processing: RTSP streams (multiple sources at once);
- capacity processing: directory of files (FPS).

Demonstrated adapters:
- RTSP source adapter;
- video file source adapter;
- Always-ON RTSP sink adapter;
- Video/Metadata sink adapter.


**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/background_remover
git lfs pull

# if you want to share with us where are you from
# run the following command, it is completely optional
curl --silent -O -- https://hello.savant.video/peoplenet.html

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:8554/road-traffic-processed' in your player
# or visit 'http://127.0.0.1:8888/road-traffic-processed/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```
