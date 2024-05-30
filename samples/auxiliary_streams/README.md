# Auxiliary Streams

**Note**: This sample is not supported on Jetson.

A pipeline demonstrating the use of Auxiliary Streams in Savant. The pipeline contains element, [Multiple Resolutions](multiple_resolutions.py). It scales the frame to multiple resolution and sends the frames to the auxiliary streams.

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime configuration [guide](https://insight-platform.github.io/Savant/develop/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/auxiliary_streams/docker-compose.x86.yml up

# open 'rtsp://127.0.0.1:554/stream/video-360p', 'rtsp://127.0.0.1:554/stream/video-480p',
# 'rtsp://127.0.0.1:554/stream/video-720p' in your player
# or visit 'http://127.0.0.1:554/stream/video-360p', 'http://127.0.0.1:554/stream/video-480p', 
# 'http://127.0.0.1:554/stream/video-720p' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```