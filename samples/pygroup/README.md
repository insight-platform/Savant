# PyGroup: Colocating Multiple PyFuncs

A pipeline demonstrating the `pygroup` element, which colocates multiple
sequential PyFuncs into a single GStreamer element. The colocated PyFuncs run
one after another on every frame, without queues in between, while each keeps
its own OpenTelemetry span so the stages can still be profiled individually.

The sample colocates two overlay PyFuncs defined in [overlays.py](overlays.py):

- `HorizontalLineOverlay` ("Step 1") draws a horizontal line and a label on the
  main frame, and feeds a `-h` auxiliary stream.
- `VerticalLineOverlay` ("Step 2") draws a vertical line and a label on top of
  Step 1's result — visibly confirming the group runs sequentially — and feeds
  a `-v` auxiliary stream.

Both stages are declared under a single `pygroup` unit in
[module.yml](module.yml). The bundle also ships a preconfigured Jaeger/OTLP
setup so you can inspect the per-stage spans.

Tested on platforms:

- Nvidia Turing, Ampere
- Nvidia Jetson Orin family

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime configuration [guide](https://insight-platform.github.io/Savant/develop/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/pygroup/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/pygroup/docker-compose.l4t.yml up

# open the main stream and the two auxiliary streams in your player:
#   'rtsp://127.0.0.1:554/stream/video'   (main frame, both overlays)
#   'rtsp://127.0.0.1:554/stream/video-h' (Step 1 auxiliary stream)
#   'rtsp://127.0.0.1:554/stream/video-v' (Step 2 auxiliary stream)
# or visit 'http://127.0.0.1:888/stream/video' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

## Inspect the Telemetry

The bundle runs an all-in-one Jaeger instance and configures the module to
export traces to it over OTLP. Open the Jaeger UI to see, per frame, a
`process-frame` span with one nested span per colocated PyFunc
(`samples.pygroup.overlays.HorizontalLineOverlay` and
`samples.pygroup.overlays.VerticalLineOverlay`), each of which further nests its
own `draw-*` and `aux-stream-publish` spans:

```
http://127.0.0.1:16686
```

**Note**: On x86 the sample encodes output with H.264. The Jetson
(`docker-compose.l4t.yml`) bundle uses the JPEG codec instead, because
entry-level devices such as the Jetson Orin Nano do not provide an NVENC
hardware encoder.

See the [Python Function Unit](https://insight-platform.github.io/Savant/develop/savant_101/70_python.html)
documentation for more details on the `pygroup` element.
