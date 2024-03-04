# Conditional Video Processing

A simple pipeline that demonstrates conditional video processing, drawing on frames and encoding. The first element of the pipeline is the pyfunc [ConditionalSkipProcessing](conditional_video_processing.py). Pyfunc checks if the source should be processed by checking the value of the corresponding parameter (the source name) in Etcd. If not, it removes the primary object and therefore disables downstream inference. The secondary element is the [Nvidia PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) model. The model is used to detect persons in the video. When the model detects person, the pyfunc [ConditionalVideoProcessing](conditional_video_processing.py) adds tags `draw` and `encode` to the frame. DrawFunc draws on frame only when tag `draw` is present and encoder encodes frame only when tag `encode` is present. On Always-On RTSP sink you can see the video constantly switching between original and stub frames.

Preview:

![](assets/conditional-video-processing.webp)

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime configuration [guide](https://insight-platform.github.io/Savant/develop/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

## Build Engines

The demo uses models that are compiled into TensorRT engines the first time the demo is run. This takes time. Optionally, you can prepare the engines before running the demo by using the command:

```bash
# you are expected to be in Savant/ directory

./scripts/run_module.py --build-engines samples/peoplenet_detector/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/conditional_video_processing/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/conditional_video_processing/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/city-traffic' in your player
# or visit 'http://127.0.0.1:888/stream/city-traffic/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

## Source Processing Control

By default, the pipeline processes any source. Etcd is used to control the processing. By changing the value of the key `savant/sources/{source-id}` in Etcd you can enable or disable processing of the corresponding source.

To enable/disable source processing it is convenient to use the script:

```bash
# you are expected to be in Savant/ directory

./samples/conditional_video_processing/source-switch.sh on
# or
./samples/conditional_video_processing/source-switch.sh off
```
