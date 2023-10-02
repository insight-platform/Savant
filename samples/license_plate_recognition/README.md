# License plate recognition 

The app partially reproduces [deepstream_lpr_app](https://github.com/NVIDIA-AI-IOT/deepstream_lpr_app) in the Savant framework. The pipeline detects cars using YoloV8 models and detects license plate using NVidia LPD model. Cars and plates track using NVidia traker the license plate is recognized using the NVidia LPR model. The results are displayed on the frames.

Preview:

![](assets/license-plate-recognition-1080.webp)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated adapters:

- RTSP source adapter;
- Always-ON RTSP sink adapter.

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/license_plate_recognition

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

## Performance Measurement

Download the video file to your local folder. For example, create a data folder and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/lpr_test_1080p.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/lpr_test_1080p.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
docker run --rm -it --gpus=all \
-v `pwd`/samples:/opt/savant/samples \
-v `pwd`/data:/data:ro \
-v `pwd`/models/license_plate_recognition:/models \
-v `pwd`/downloads/license_plate_recognition:/downloads \
license_plate_recognition-module \
samples/license_plate_recognition/module_performance.yml
```

or for Jetson

```bash
docker run --rm -it --runtime=nvidia \
-v `pwd`/samples:/opt/savant/samples \
-v `pwd`/data:/data:ro \
-v `pwd`/models/license_plate_recognition:/models \
-v `pwd`/downloads/license_plate_recognition:/downloads \
license_plate_recognition-module \
samples/nvidia_car_classification/module_performance.yml
```
