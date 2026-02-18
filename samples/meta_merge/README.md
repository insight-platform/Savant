# Meta Merge Sample

This sample demonstrates the Meta Merge service for merging metadata from multiple parallel inference pipelines into a single stream.

## Architecture

The pipeline topology:

```
video-loop-source → router → module-left  ─┐
                      └─────→ module-right ─┼→ meta-merge → visualization → always-on-sink (RTSP)
```

1. **Video Loop Source**: Loops a video file and publishes to the router
2. **Router**: Accepts the stream, creates left and right ROI objects on each VideoFrame (using savant-rs API), and fans out to two egress endpoints
3. **Module (2 instances)**: Each instance runs YOLO11 only on its ROI (left or right half via `input.object: router.{left|right}_roi`), outputs full frame with video copy (no draw)
4. **Meta Merge**: Merges objects from both module instances using `export_complete_object_trees` and `import_object_trees`
5. **Visualization**: Draws bounding boxes for all objects
6. **Always-On RTSP Sink**: Streams the result via RTSP

## File Layout

All components live under `src/`:

```
src/
├── router/          # Router config and pass-through handler
├── meta_merge/     # Meta-merge config and merge handler
├── detector/       # YOLO detector module with ROI filter
└── visualization/  # Overlay module
```

## Running the Sample

### x86 (with NVIDIA GPU)

```bash
docker compose -f samples/meta_merge/docker-compose.x86.yml up
```

### L4T (Jetson)

```bash
docker compose -f samples/meta_merge/docker-compose.l4t.yml up
```

## Viewing the Stream

Connect to the RTSP stream:

- **RTSP**: `rtsp://localhost:554/stream`
- **HLS**: `http://localhost:888/stream.m3u8`

## Configuration

- **MODULE_ROI**: Environment variable (`left` or `right`) determines which half of the frame each module instance processes
- **VIDEO_LOCATION**: URL or path to the video file (default: shuffle_dance demo)
- **CODEC**: `copy` for detector modules (video frame pass-through), `jpeg` for visualization output
