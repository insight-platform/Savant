# Car Detection and Classification (Nvidia detectors and classifiers, Nvidia tracker)

The app reproduces [deepstream-test2 app](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/apps/deepstream-test2) in the Savant framework. The pipeline detects and tracks cars, and applies car type, color and make classification models to detected cars. The results are displayed on the frames with bounding boxes, track ids and classification labels.

Preview:

![](../nvidia-car-classification-loop.webp)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated adapters:

- RTSP source adapter;
- Always-ON RTSP sink adapter.

A step-by-step [tutorial](https://hello.savant.video/nvidia-test2).

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/nvidia_car_classification

# if you want to share with us where are you from
# run the following command, it is completely optional
curl --silent -O -- https://hello.savant.video/cars.html

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:8554/nvidia-sample-processed' in your player
# or visit 'http://127.0.0.1:8888/nvidia-sample-processed/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```
