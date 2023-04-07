# OpenCV CUDA MOG2 Background Segmentation Demo

A simple pipeline demonstrates removing the background with accelerated MOG2 from OpenCV CUDA. The pipeline takes less than 2 ms to decode, remove the background, draw the background, and encode the frame to HEVC, which results in achieving 500+ FPS on modern hardware.

The left side represents the orignial frame; the right side represents the framw without the background.

Preview:

![](../opencv_cuda_bg_remover_mog2.webp)

Features:

![demo-0 2 1-mog2 (2)](https://user-images.githubusercontent.com/15047882/230607388-febddb9b-c5da-417d-a563-4a56829c82ab.png)

YouTube Video:

[![Watch the video](https://img.youtube.com/vi/P9w-WS6HLew/default.jpg)](https://youtu.be/P9w-WS6HLew)

A step-by-step [tutorial](https://hello.savant.video/opencv-cuda-bg-remover-mog2-tutorial).

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated operational modes:

- real-time processing: RTSP streams (multiple sources at once);
- capacity processing: directory of files (FPS);
- classical computer vision algorithms (OpenCV CUDA).

Demonstrated adapters:
- RTSP source adapter;
- Always-ON RTSP sink adapter;

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/opencv_cuda_bg_remover_mog2
git lfs pull

# if you want to share with us where are you from
# run the following command, it is completely optional
curl --silent -O -- https://hello.savant.video/opencv_cuda_bg_remover_mog2.html

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


## Performance Measurement

Download the video file to your local folder. For example, create a data folder and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/road_traffic.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/road_traffic.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
docker run --rm -it --gpus=all \
-v `pwd`/samples:/opt/app/samples \
-v `pwd`/data:/data:ro \
ghcr.io/insight-platform/savant-deepstream:0.2.0-6.2-samples \
samples/opencv_cuda_bg_remover_mog2/demo_performance.yml
```

or for Jetson

```bash
docker run --rm -it --gpus=all \
-v `pwd`/samples:/opt/app/samples \
-v `pwd`/data:/data:ro \
ghcr.io/insight-platform/savant-deepstream-l4t:0.2.0-6.2-samples \
samples/opencv_cuda_bg_remover_mog2/demo_performance.yml
```
