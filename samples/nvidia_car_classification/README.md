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

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/nvidia_car_classification

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

mkdir -p data && curl -o data/deepstream_sample_720p.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/deepstream_sample_720p.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/nvidia_car_classification/run_perf.sh
```
