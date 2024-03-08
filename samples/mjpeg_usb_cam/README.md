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

# Stream control API for Always-On-RTSP sink usage examples

The Always-On-RTSP sink has a control API to start and stop the stream. The API is available at http://localhost:13000. The API is documented is available at http://localhost:13000/docs.

## Dump all streams configuration

Dump all streams configuration in JSON format:

```bash
curl 'http://localhost:13000/streams?format=json'
```

Dump all streams configuration in YAML format:

```bash
curl 'http://localhost:13000/streams?format=yaml'
```

## Stop and delete a stream

```bash
curl -X DELETE 'http://localhost:13000/streams/gige-raw'
```

## Create and start a stream

```bash
curl -X PUT \
    -H 'Content-Type: application/json' \
    -d '{"stub_file":"/stub_imgs/smpte100_1280x720.jpeg","framerate":"20/1","bitrate":4000000,"profile":"High","codec":"h264","max_delay_ms":1000,"latency_ms":null,"transfer_mode":"scale-to-fit","rtsp_keep_alive":true,"metadata_output":null,"sync_output":false}' \
    'http://localhost:13000/streams/gige-raw'
```
