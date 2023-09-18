# Multiple GigE Vision Cameras Demo

A simple pipeline demonstrates how GigE Vision Source Adapter works in Savant. In the demo video from one GigE Vision camera is passed as raw-rgba frames, and another one is passed as HEVC-encoded frames. Both streams are passed to an Always-On-RTSP sink.

The resulting streams can be accessed via LL-HLS on `http://locahost:888/stream/gige-raw` (raw-rgba frames) and `http://locahost:888/stream/gige-encoded` (HEVC-encoded frames).

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/multiple_gige
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# visit 'http://127.0.0.1:888/stream/gige-raw' and 'http://127.0.0.1:888/stream/gige-encoded' to see how it works

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```
