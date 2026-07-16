# Scene change

The pipeline computes a ReID embedding vector for a region of interest (ROI) on the frame and compares it with a vector of a frame N seconds ago to detect a scene change (e.g., when a camera moved). The ROI is configurable per-source in Etcd.

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

./scripts/run_module.py --build-engines samples/scene_change/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/scene_change/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/scene_change/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/video' in your player
# or visit 'http://127.0.0.1:888/stream/video/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

## ROI configuration

By default, the pipeline uses ROI from the [module configuration](module.yml). By changing the value of the key `savant/roi/{source_id}` in Etcd you can change ROI of the corresponding source.

To change a source ROI it is convenient to use the script:

```bash
# you are expected to be in Savant/ directory
./samples/scene_change/set-roi.sh video "540,100,1000,400"
# to reset to the default ROI
./samples/scene_change/set-roi.sh video
```
