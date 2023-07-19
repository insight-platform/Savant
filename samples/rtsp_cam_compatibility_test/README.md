# Simple Pipeline For Testing RTSP Camera Compatibility

This pipeline is a simple pipeline that can be used to test the compatibility of RTSP cameras with Savant. It takes an RTSP stream as input and outputs the stream to HLS. The pipeline can be used to test the compatibility of RTSP cameras with Savant.

It uses NVDEC and NVENC internally and Savant protocol. Thus, if the pipeline works, it means that Savant highly likely will work with the camera.

The resulting video is broadcast in 640x360 resolution. You can access it at `http://<ip>:888/stream`.

## Specifying the RTSP URL

Edit `.env` file and set the `URI` variable to the RTSP URL of the camera.

## X86

```bash
docker-compose -f docker-compose.x86.yml up
```

# L4T (Jetson)

```bash
docker-compose -f docker-compose.l4t.yml up
```
