# People detection, tracking and face blurring (PeopleNet, Nvidia Tracker, OpenCV CUDA)

A simple pipeline that uses
standard [Nvidia PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) model to detect
persons and their faces in the video. The faces are matched versus bodies and blurred with the integrated OpenCV CUDA
functionality. There is also a simple unreliable tracker that helps reduce flickering of boxes.

GPU-accelerated blurring made with OpenCV is a killer feature of this demonstration. It enables very fast and efficient
face blurring without CPU utilization.

The **Green** Icon represents how many people with blurred faces in the scene.
The **Blue** Icon represents how many people with blurred faces in the scene.

Preview:

![](assets/peoplenet-blur-demo-loop.webp)

Features:

![demo-0 2 0-dia](https://user-images.githubusercontent.com/15047882/229354155-f37ad24b-c0bd-4249-ba83-a53b3c1e053b.png)

YouTube Video:

[![Watch the video](https://img.youtube.com/vi/rHAY9SKONRQ/default.jpg)](https://youtu.be/rHAY9SKONRQ)

A
step-by-step [tutorial](https://blog.savant-ai.io/meet-savant-a-new-high-performance-python-video-analytics-framework-for-nvidia-hardware-22cc830ead4d?source=friends_link&sk=c7169b378b31451ab8de3d882c22a774).

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

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime
configuration [guide](https://insight-platform.github.io/Savant/develop/getting_started/0_configure_prod_env.html) helps
to configure the runtime to run Savant pipelines.

## Build Engines

The demo uses models that are compiled into TensorRT engines the first time the demo is run. This takes time.
Optionally, you can prepare the engines before running the demo by using the command:

```bash
# you are expected to be in Savant/ directory

./scripts/run_module.py --build-engines samples/peoplenet_detector/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/peoplenet_detector/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/peoplenet_detector/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/city-traffic' in your player
# or visit 'http://127.0.0.1:888/stream/city-traffic/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

## Performance Measurement

Download the video file to the data folder. For example:

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/Free_City_Street_Footage.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/Free_City_Street_Footage.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/peoplenet_detector/run_perf.sh
```

**Note**: Change the value of the `DATA_LOCATION` variable in the `run_perf.sh` script if you changed the video.
