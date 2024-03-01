# Intersection traffic meter demo

**NB**: The demo uses **YOLOV8** model which takes up to **10-15 minutes** to compile to TensorRT engine. The first launch may take a decent time.

The pipeline detects when cars, trucks or buses cross a city intersection delimited by user-configured polygon and the direction of the crossing. The crossing events are attached to individual tracks and are counted for each video source and polygon edge separately; the counters are displayed on the frame. The crossing events are also stored with Graphite and displayed on a Grafana dashboard.

Preview:

![](assets/intersection-traffic-meter-loop.webp)

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

./samples/intersection_traffic_meter/build_engines.sh
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/intersection_traffic_meter/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/intersection_traffic_meter/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/leeds' in your player
# or visit 'http://127.0.0.1:888/stream/leeds/' (LL-HLS)

# for pre-configured Grafana dashboard visit
# http://127.0.0.1:3000/d/WM6WimE4z/crossings?orgId=1&refresh=5s

# Ctrl+C to stop running the compose bundle
```

To create a custom Grafana dashboard, sign in with `admin\admin` credentials.

## Detector Model

This demo uses [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo/) to process YOLO models and allows you to use any YOLO model that DeepStream-Yolo supports. The demo uses an already prepared `yolov8m`. If you are going to use any other model (e.g. custom yolov8m), follow the DeepStream-Yolo export instructions. For example, YOLOv8 instructions are [here](https://github.com/marcoslucianops/DeepStream-Yolo/blob/master/docs/YOLOv8.md).

## Performance Measurement

Download the video file to the data folder. For example:

```bash
# you are expected to be in Savant/ directory

mkdir -p data
curl -o data/leeds_1080p.mp4 https://eu-central-1.linodeobjects.com/savant-data/demo/leeds_1080p.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/intersection_traffic_meter/run_perf.sh
```

**Note**: Change the value of the `DATA_LOCATION` variable in the `run_perf.sh` script if you changed the video.
