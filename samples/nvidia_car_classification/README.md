# Car Detection and Classification (Nvidia detectors and classifiers, Nvidia tracker)

The app reproduces [deepstream-test2 app](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test2) in the Savant framework. The pipeline detects and tracks cars, and applies car type, color and make classification models to detected cars. The results are displayed on the frames with bounding boxes, track ids and classification labels.

Preview:

![](assets/nvidia-car-classification-loop.webp)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated adapters:

- RTSP source adapter;
- Always-ON RTSP sink adapter.

A step-by-step [tutorial](https://blog.savant-ai.io/building-a-high-performance-car-classifier-pipeline-with-savant-b232461ad96?source=friends_link&sk=63cb289315679af83032ef5247861a2d).

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

./scripts/run_module.py --build-engines samples/nvidia_car_classification/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/nvidia_car_classification/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/nvidia_car_classification/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/nvidia-sample-processed' in your player
# or visit 'http://127.0.0.1:888/stream/nvidia-sample-processed/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

## Performance Measurement

Download the video file to the data folder. For example:

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/deepstream_sample_720p.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/deepstream_sample_720p.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/nvidia_car_classification/run_perf.sh
```

**Note**: Change the value of the `DATA_LOCATION` variable in the `run_perf.sh` script if you changed the video.
