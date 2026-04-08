# NvArgusCameraSrc Demo

A simple pipeline demonstrating how to use `nvarguscamerasrc_bin` as a source element in Savant for capturing video from CSI cameras on NVIDIA Jetson (L4T) platforms. The `nvarguscamerasrc_bin` element is a custom GStreamer Bin that wraps one or more `nvarguscamerasrc` elements, allowing multiple CSI cameras to be used as a single pipeline source.

Each camera source is independently configurable with its own:

- `source-id` — identifier for the stream in the pipeline
- `framerate` — capture framerate
- `properties` — per-element `nvarguscamerasrc` properties (e.g., `sensor-id`, `sensor-mode`)

This demo captures video from two CSI cameras and streams the output to an Always-On RTSP sink without any processing.

Tested on platforms:

- NVIDIA Jetson (L4T)

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: This sample requires an NVIDIA Jetson device with one or more CSI cameras connected.

Ensure the `nvargus-daemon` service is running:

```bash
sudo systemctl start nvargus-daemon
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# Run the demo
docker compose -f samples/nvarguscamerasrc/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/argus-0' in your player
# or visit 'http://127.0.0.1:888/stream/argus-0/' (LL-HLS)
#
# for the second camera:
# open 'rtsp://127.0.0.1:554/stream/argus-1'
# or visit 'http://127.0.0.1:888/stream/argus-1/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```
