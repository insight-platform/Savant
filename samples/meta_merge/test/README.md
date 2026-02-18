# Meta-merge Test Framework

Source → infer_blackbox → sink topology for predictable pipeline testing.

## Prerequisites

- Docker and Docker Compose
- Test image: `test_image.jpeg` (created automatically by `create_test_image.py` if missing)

## Run the Test

```bash
cd Savant/samples/meta_merge/test

# Create test image if missing
python create_test_image.py

# Build (first run downloads YOLO model; takes a few minutes)
docker compose -f docker-compose.test.yml build

# Run
docker compose -f docker-compose.test.yml up --abort-on-container-exit
```

The sink exits with code 0 on success (all detections matched expected IoU), 1 on failure. All three services (source, infer, sink) should exit 0 when the test passes.

To show only the test result line:

```bash
docker compose -f docker-compose.test.yml up --abort-on-container-exit 2>&1 | grep -E 'PASSED|FAILED'
```

## Services

- **source**: Loads JPEG, merges L|R, runs YOLOv11, stores (source, person) attribute, sends via BlockingWriter
- **infer**: BlockingReader/Writer, creates ROIs, runs YOLO, attaches person detections to ROIs, forwards to sink
- **sink**: BlockingReader, compares detected objects with (source, person) using IoU, reports pass/fail

## Environment

| Service | Variable | Default | Description |
|---------|----------|---------|-------------|
| source | ZMQ_SOCKET | dealer+connect:ipc:///tmp/zmq-sockets/infer.ipc | Writer endpoint |
| source | REPETITIONS | 1 | Number of frames to send |
| source | FRAME_INTERVAL_MS | 0 | Pause between frame sends (ms), 0 = no delay |
| source | IMAGE_PATH | /app/test_image.jpeg | Input image path |
| infer | ZMQ_INGRESS | router+bind:ipc:///tmp/zmq-sockets/infer.ipc | Reader endpoint |
| infer | ZMQ_EGRESS | dealer+bind:ipc:///tmp/zmq-sockets/sink.ipc | Writer endpoint |
| sink | ZMQ_SOCKET | router+connect:ipc:///tmp/zmq-sockets/sink.ipc | Reader endpoint |
| all | ZMQ_SEND_TIMEOUT_MS | 10000 | Writer send timeout (ms) |
| all | ZMQ_RECEIVE_TIMEOUT_MS | 5000 | Reader/Writer receive timeout (ms) |
| all | ZMQ_SEND_RETRIES | 10 | Writer send retries |
| all | ZMQ_RECEIVE_RETRIES | 10 | Writer receive (ack) retries |

ZMQ timeout/retry env vars reduce "Resource temporarily unavailable" warnings during startup and EOS handshake.
