# Simple Pipeline For Testing RTSP Camera Compatibility

This pipeline is a simple pipeline that can be used to test the compatibility of RTSP cameras with Savant. It takes an RTSP stream as input and outputs the stream to HLS. The pipeline can be used to test the compatibility of RTSP cameras with Savant.

It uses NVDEC and NVENC internally and Savant protocol. Thus, if the pipeline works, it means that Savant highly likely will work with the camera.

The resulting video is broadcast in 640x360 resolution. You can access it at `http://<ip>:888/stream/test`.

Tested on platforms:

- Nvidia Turing
- Nvidia Jetson Orin family

## RTSP Adapter Variants

The sample allows testing RTSP streams with:

- FFmpeg-based RTSP adapter (does not support RTCP Sender Reports, but potentially more cameras are supported);
- Retina-based RTSP adapter (supports RTCP Sender Reports and cross-stream synchronizations, but potentially fewer cameras are supported).


## Specifying the RTSP URL

Edit `.env` file and set the `URI` variable to the RTSP URL of the camera.

Example: 

```
URI=rtsp://hello.savant.video:8554/stream/town-centre
```

### RTSP Credentials

For FFmpeg encode login and password in the URI. For Retina RTSP use the `RETINA_RTSP_CREDENTIALS` variable in the `.env` file.

Example:

```
RETINA_RTSP_CREDENTIALS={"username": "admin", "password": "password"}
```

## FFmpeg adapter

```bash
docker compose -f samples/rtsp_cam_compatibility_test/docker-compose-retina.yml up
```

See if it works: http://127.0.0.1:888/stream/test

# Retina adapter

```bash
docker compose -f samples/rtsp_cam_compatibility_test/docker-compose-retina.yml up
```

See if it works:

- Stream without RTCP SR from camera: http://127.0.0.1:888/stream/no-rtcp-sr/ (it will work if camera is supported, otherwise stub)
- Stream with RTCP SR from camera: http://127.0.0.1:888/stream/rtcp-sr/ (it will work if camera is supported and sends RTCP SR, otherwise stub)
