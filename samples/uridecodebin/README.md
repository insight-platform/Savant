# URIDecodeBin Source Demo

A simple pipeline demonstrating how to use `uridecodebin` as a source element in Savant. The `uridecodebin` element is a versatile GStreamer component that automatically handles decoding from various URI schemes including:

- Local files (`file:///path/to/video.mp4`)
- RTSP streams (`rtsp://...`)
- HTTP/HTTPS streams (`http://...`, `https://...`)
- And other URI-based sources

This demo ingests video from a configurable URI and streams it to an Always-On RTSP sink without any processing. The pipeline includes an egress frame filter that sets the source ID for the output stream.

Tested on platforms:

- Nvidia Turing, Ampere

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime configuration [guide](https://insight-platform.github.io/Savant/develop/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

## Run Demo

Before running the demo, you need to specify the URI of the video source. Set the `URI` environment variable to point to your video source. Use `.env` file for that.

```bash
# you are expected to be in Savant/ directory

# .env contains this download uri by default
#
mkdir -p data && curl -o data/road_traffic.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/road_traffic.mp4

# Run the demo
docker compose -f samples/uridecodebin/docker-compose.yml up

# open 'rtsp://127.0.0.1:554/stream/video-source-1' in your player
# or visit 'http://127.0.0.1:888/stream/video-source-1/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```
