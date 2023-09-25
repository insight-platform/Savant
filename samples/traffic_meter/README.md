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

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/traffic_meter
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# for pre-configured Grafana dashboard visit
# http://127.0.0.1:3000/d/WM6WimE4z/entries-exits?orgId=1&refresh=5s

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

To create a custom Grafana dashboard, sign in with `admin\admin` credentials.

## Switch detector model

The sample includes an option to choose the model used for object detection. Choose between NVIDIA [peoplenet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) and YOLOv8 by changing the env variable in `.env` file:

- `DETECTOR=peoplenet` for peoplnet
- `DETECTOR=yolov8m` for yolov8m
- `DETECTOR=yolov8s` for yolov8s

## Performance Measurement

Download the video file to your local folder. For example, create a data folder and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/AVG-TownCentre.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/AVG-TownCentre.mp4
```

Next, if you haven't run the sample in the default mode yet (following the instructions above), run

```bash
docker compose -f samples/traffic_meter/docker-compose.x86.yml build module
```

or

```bash
docker compose -f samples/traffic_meter/docker-compose.l4t.yml build module
```

to build the module docker image.

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/traffic_meter/run_perf.sh
```

Note `-e DETECTOR=yolov8m` is set by default.
