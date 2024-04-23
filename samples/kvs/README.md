# KVS Sample

A pipeline demonstrates how to send frames to and receive from Kinesis Video Stream. Pipeline consists of two parts: exporter and importer. Exporter processes frames from a video file, sends metadata to MongoDB and sends frames to Kinesis Video Stream. Importer receives frames from Kinesis Video Stream, retrieves metadata from MongoDB and draw bboxes on frames.

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

./scripts/run_module.py --build-engines samples/peoplenet_detector/module.yml
```

## Run Demo

Before running the demo, set AWS credentials in the [samples/kvs/.env](.env) file.

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/kvs/docker-compose.x86.yml up --build

# if Jetson
docker compose -f samples/kvs/docker-compose.l4t.yml up --build

# Wait for 1 minute after kvs-sink starts to send frames to Kinesis Video Stream:
# in kvs-sink logs, you will see "Creating Kinesis Video Client" message.
# Then send "play" command to kvs-source to start receiving frames from Kinesis Video Stream

curl -X PUT http://localhost:18367/stream/play

# open 'rtsp://127.0.0.1:554/stream/video' in your player
# or visit 'http://127.0.0.1:888/stream/video/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```
