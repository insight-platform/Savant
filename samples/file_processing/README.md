# File Processing Demo

A simple pipeline demonstrates how to handle files (input/output). 

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

./scripts/run_module.py --build-engines samples/file_processing/module.yml
```

## Launch the pipeline and video file sink

```bash
docker compose -f samples/file_processing/docker-compose.x86.yml up
```

## Ingest a file

Known issues and how to fix them: https://github.com/insight-platform/Savant/issues/1140

Place a file in `data/your-video.mp4`.

```
docker compose \
    -f samples/file_processing/docker-compose.x86.yml \
    --profile video-file-input \
    run --rm \
    -e LOCATION="/data/your-video.mp4" \
    -e SOURCE_ID=your-source \
    video-file-input
```

Check results in `data/results/your-source/your-video/video.mov`.