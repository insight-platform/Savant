# MJPEG USB Camera Access

A pipeline demonstrating how to capture MJPEG from a USB camera. MJPEG is a common format for USB/MIPI CSI-2 cameras providing compressed, low-latency video streaming.

The resulting stream can be accessed via LL-HLS on `http://locahost:888/stream/video`

## Hardware Acceleration Notes

On X86, JPEG decoding and encoding is done in software (or hardware-assisted if dGPU and drivers support it). On Jetson, JPEG decoding and decoding is done in hardware with NVJPEG.

To safely work on Jetson Orin Nano, the AO-RTSP adapter is configured without NVIDIA hardware acceleration, if you are **not** on Orin Nano, you can pass extra configuration to enable the Nvidia-accelerated runtime with:

```yaml
  always-on-sink:
    runtime: nvidia
```

## How To Run The Demo

Edit the `docker-compose.x86.yml` or `docker-compose.l4t.yml` file to configure your USB-camera properly. The parameters you need to fix are displayed below:

```yaml
    environment:
      ...
      - FFMPEG_PARAMS=input_format=mjpeg,video_size=1920x1080
    devices:
      - /dev/video0:/dev/video0

```

```bash
# if x86
docker compose -f samples/mjpeg_usb_cam/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/mjpeg_usb_cam/docker-compose.l4t.yml up

# visit 'http://127.0.0.1:888/stream/video' in your browser
# Ctrl+C to stop running the compose bundle
```
