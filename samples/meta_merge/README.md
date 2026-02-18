# Meta Merge Sample

![](assets/meta-merge.webp)

Demonstrates the Meta Merge service for merging metadata from multiple parallel
inference pipelines into a single stream.  A video frame is split into
left/right ROIs by a router, each half is processed independently by a YOLO
detector module, and the results are merged back into a single frame for
visualization.

## Architecture

```
                         ┌─ module-left  ─┐
video-loop-source → router                 ├→ meta-merge → visualization → always-on-sink
                         └─ module-right ─┘
```

| Service | Role |
|---------|------|
| **video-loop-source** | Loops a video file and publishes frames to the router |
| **router** | Creates left/right ROI objects on each frame, fans out to two detector modules |
| **module-left / module-right** | Runs YOLOv11n on its ROI half (`router.left_roi` / `router.right_roi`) |
| **meta-merge** | Merges object trees from both modules back onto a single frame |
| **visualization** | Draws ROI boundaries and detection bounding boxes (declarative `rendered_objects`) |
| **always-on-sink** | Streams the annotated result via RTSP/HLS |

## Directory Layout

```
meta_merge/
├── docker-compose.l4t.yml          # Base compose — Jetson (L4T) images
├── docker-compose.x86.yml          # x86 dGPU — extends l4t, swaps images
├── docker-compose.x86-test.yml     # x86 E2E test — replaces source/sink
├── README.md
├── src/
│   ├── router/
│   │   ├── config.json             # Router socket config (env-parameterised)
│   │   └── handler.py              # Creates left/right ROI objects
│   ├── detector/
│   │   ├── module.yml              # YOLOv11n detector config
│   │   ├── roi_ingress_filter.py   # Filters frames by MODULE_ROI
│   │   └── roi_resolver.py         # ${py:} resolver for ROI input object
│   ├── meta_merge/
│   │   ├── config.json             # Meta-merge socket config
│   │   ├── module.py               # Merge/ready/expired/late handlers
│   │   └── roi_constants.py        # Shared ROI namespace/label constants
│   └── visualization/
│       └── module.yml              # Declarative draw_func with rendered_objects
└── test/
    ├── Dockerfile                  # savant-rs + PyTorch + Ultralytics
    ├── source.py                   # Test source: YOLO → expected attrs → DEALER
    ├── sink.py                     # Test sink: ROUTER → IoU comparison
    ├── create_test_image.py        # Downloads a sample person image
    └── test_image.jpeg             # Test input image
```

## Running

### Prerequisites

- Docker with Compose v2
- NVIDIA GPU with drivers installed

### Demo — x86 (dGPU)

Runs the full pipeline with a looping video and RTSP output.

```bash
cd samples/meta_merge
docker compose -f docker-compose.x86.yml up
```

### Demo — Jetson (L4T)

```bash
cd samples/meta_merge
docker compose -f docker-compose.l4t.yml up
```

### End-to-End Test — x86

Replaces `video-loop-source` and `always-on-sink` with a test source and test
sink.  The source runs YOLO on a static image, stores expected person bounding
boxes as a frame attribute, and sends frames through the pipeline.  The sink
compares detected objects against the stored expectations using IoU.

```bash
cd samples/meta_merge

# Create test image if missing
python test/create_test_image.py

# Build (first run downloads the YOLO model — takes a few minutes)
docker compose -f docker-compose.x86-test.yml build

# Run (exits when the sink finishes)
docker compose -f docker-compose.x86-test.yml up --abort-on-container-exit
```

The sink exits with code **0** on PASSED, **1** on FAILED.

## Viewing the Stream

When running in demo mode the always-on-sink publishes RTSP/HLS:

| Protocol | URL |
|----------|-----|
| RTSP | `rtsp://localhost:554/stream` |
| HLS | `http://localhost:888/stream.m3u8` |
| WebRTC | `http://localhost:8889/stream` |

## Startup Order

Services start in dependency order to avoid message loss:

```
module-left ──┐                               ┌── always-on-sink ── video-loop-source
              ├→ router ── meta-merge → visualization
module-right ─┘
```

1. **module-left** and **module-right** start first (no dependencies)
2. **router** waits until both modules are **healthy** (built-in healthcheck)
3. **meta-merge** waits until both modules are **healthy**
4. **visualization** waits until meta-merge has **started**
5. **always-on-sink** waits until visualization is **healthy**
6. **video-loop-source** starts last (after sink is started)

## ZMQ Transport

All inter-service communication uses **DEALER/ROUTER** sockets over IPC:

| Link | Writer (DEALER) | Reader (ROUTER) |
|------|-----------------|-----------------|
| source → router | `dealer+connect` | `router+bind` |
| router → modules | `dealer+bind` | `router+connect` |
| modules → meta-merge | `dealer+connect` | `router+bind` |
| meta-merge → visualization | `dealer+bind` | `router+connect` |
| visualization → sink | `dealer+bind` | `router+connect` |

## Configuration

### Common

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_LOCATION` | shuffle_dance demo | Video URL or path for the source adapter |
| `LOGLEVEL` | `info` | Log verbosity (`trace`, `debug`, `info`, `error`) |

### Detector Modules

| Variable | Default | Description |
|----------|---------|-------------|
| `MODULE_ROI` | `left` | Which ROI half this instance processes (`left` / `right`) |
| `CODEC` | `copy` | Frame codec — `copy` passes through without re-encoding |

### Visualization

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEC` | `jpeg` (L4T) / `h264` (x86) | Output encoding for the sink |

### E2E Test Source

| Variable | Default | Description |
|----------|---------|-------------|
| `REPETITIONS` | `10` | Number of frames to send |
| `FRAME_INTERVAL_MS` | `10` | Pause between sends (ms) |
| `POST_EOS_IDLE_S` | `3600` | Idle after EOS (keeps container alive while pipeline drains) |
| `FRAME_WIDTH` | `1280` | Target frame width |
| `FRAME_HEIGHT` | `720` | Target frame height |

### E2E Test Sink

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_STARTUP_S` | `600` | Max wait for first frame before failing |
| `MAX_IDLE_S` | `60` | Max idle after last frame before declaring done |
