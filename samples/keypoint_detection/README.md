# Keypoint detection demo

This application demonstrates the human body key point detection model.

Preview:

![](assets/shuffle_dance.webp)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated adapters:

- Video loop source adapter;
- Always-ON RTSP sink adapter.

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

./scripts/run_module.py --build-engines samples/keypoint_detection/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/keypoint_detection/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/keypoint_detection/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/video' in your player
# or visit 'http://127.0.0.1:888/stream/video/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```
