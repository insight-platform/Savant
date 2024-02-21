# License plate recognition 

The app partially reproduces [deepstream_lpr_app](https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app) in the Savant framework. The pipeline detects cars using YoloV8 models and detects license plate using NVidia LPD model. Cars and plates track using NVidia traker the license plate is recognized using the NVidia LPR model. The results are displayed on the frames.

Preview:

![](assets/license-plate-recognition.webp)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated adapters:

- Video loop source adapter;
- Always-ON RTSP sink adapter.

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

./scripts/run_module.py --build-engines samples/license_plate_recognition/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/license_plate_recognition/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/license_plate_recognition/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

## Performance Measurement

Download the video file to the data folder. For example:

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/lpr_test_1080p.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/lpr_test_1080p.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/license_plate_recognition/run_perf.sh
```

**Note**: Change the value of the `DATA_LOCATION` variable in the `run_perf.sh` script if you changed the video.
