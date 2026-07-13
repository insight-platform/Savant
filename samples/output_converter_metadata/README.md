# Per-source Converter Configuration from Etcd

A simple pipeline demonstrates how metadata processing in output converters works in Savant. In the demo, two RTSP streams are ingested in the module and processed with the PeopleNet model. The output converter is configurable via etcd.

The resulting streams can be accessed via LL-HLS on `http://locahost:888/stream/city-traffic` and `http://locahost:888/stream/town-centre` or via RTSP on `rtsp://127.0.0.1:554/stream/city-traffic` and `rtsp://127.0.0.1:554/stream/town-centre`.

Two RTSP streams (`city-traffic` and `town-centre`) are ingested by a single module and
processed with a YOLO11n detector. The detector's output converter is a custom subclass
of the built-in YOLO converter that, for every frame, reads the frame's `source_id` from
the converter `metadata` argument and looks up a per-source configuration object in Etcd
(e.g. the detection `confidence_threshold`). Values can be changed **live** with
`etcdctl` — no pipeline restart required.

This relies on the output-converter `metadata` argument: when a converter's `__call__`
declares a `metadata` parameter it receives the frame's `NvDsFrameMeta` wrapper
(`source_id`, `pts`, `video_frame`, objects, tags). Converters that do not declare it keep
working unchanged. See `samples/output_converter_metadata/converter.py`.

The resulting streams can be accessed via LL-HLS on
`http://localhost:888/stream/city-traffic` and `http://localhost:888/stream/town-centre`.

Tested on platforms:

- Nvidia Ampere

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime configuration [guide](https://insight-platform.github.io/Savant/develop/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

## Build Engines

The demo uses models that are compiled into TensorRT engines the first time the demo is run. This takes time. Optionally, you can prepare the engines before running the demo by using the command:

```bash
# you are expected to be in Savant/ directory

./scripts/run_module.py --build-engines samples/output_converter_metadata/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/output_converter_metadata/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/output_converter_metadata/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/city-traffic' in your player
# or visit 'http://127.0.0.1:888/stream/city-traffic' (LL-HLS)

# open 'rtsp://127.0.0.1:554/stream/town-centre' in your player
# or visit 'http://127.0.0.1:888/stream/town-centre' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

## Per-source Configuration

The converter reads the Etcd key `savant/source/<source_id>` as a JSON object.
Supported fields: `confidence_threshold` and `nms_iou_threshold`. Use the helper script to
set or update a source's configuration (the `etcd` service must be running):

```bash
# you are expected to be in Savant/samples/output_converter_metadata/ directory

# keep low-confidence detections on city-traffic (more boxes)
./set-config.sh city-traffic '{"confidence_threshold": 0.2}'

# require high confidence on town-centre (fewer boxes)
./set-config.sh town-centre '{"confidence_threshold": 0.7}'
```
