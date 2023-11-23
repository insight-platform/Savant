# Traffic meter demo


**NB**: The demo optionally uses **YOLOV8** model which takes up to **10-15 minutes** to compile to TensorRT engine. The first launch may take a decent time.

The pipeline detects when people cross a user-configured line and the direction of the crossing. The crossing events are attached to individual tracks, counted for each source separately and the counters are displayed on the frame. The crossing events are also stored with Graphite and displayed on a Grafana dashboard.

Pedestrians preview:

![](assets/traffic-meter-loop.webp)

Vehicles preview:

![](assets/road-traffic-loop.webp)

Article on Medium: [Link](https://blog.savant-ai.io/efficient-city-traffic-metering-with-peoplenet-yolov8-savant-and-grafana-at-scale-d6f162afe883?source=friends_link&sk=ab96c5ef3c173902559f213849dede9b)

Tested on platforms:

- Xavier NX, Xavier AGX;
- Nvidia Turing, Ampere.

Demonstrated operational modes:

- real-time processing: RTSP streams (multiple sources at once);

Demonstrated adapters:
- Video loop adapter;
- Always-ON RTSP sink adapter;

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

./samples/traffic_meter/build_engines.sh
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/traffic_meter/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/traffic_meter/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# for pre-configured Grafana dashboard visit
# http://127.0.0.1:3000/d/WM6WimE4z/entries-exits?orgId=1&refresh=5s

# Ctrl+C to stop running the compose bundle
```

To create a custom Grafana dashboard, sign in with `admin\admin` credentials.

## Switch Detector Model

The sample includes an option to choose the model used for object detection. Choose between NVIDIA [peoplenet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) and YOLOv8 by changing the env variable in `.env` file:

- `DETECTOR=peoplenet` for peoplnet
- `DETECTOR=yolov8m` for yolov8m
- `DETECTOR=yolov8s` for yolov8s

This demo uses [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo/) to process YOLO models and allows you to use any YOLO model that DeepStream-Yolo supports. The demo contains 2 already prepared models - `yolov8m` / `yolov8s`. If you are going to use any other model (e.g. custom yolov8m), follow the DeepStream-Yolo export instructions. For example, YOLOv8 instructions are [here](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/YOLOv8.md).

## Performance Measurement

Download the video file to the data folder. For example:

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/AVG-TownCentre.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/AVG-TownCentre.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/traffic_meter/run_perf.sh
```

**Note**: Change the value of the `DATA_LOCATION` variable in the `run_perf.sh` script if you changed the video.

**Note**: `yolov8m` detector is set by default.
