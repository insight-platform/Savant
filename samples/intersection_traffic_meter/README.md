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

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/intersection_traffic_meter
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# for pre-configured Grafana dashboard visit
# http://127.0.0.1:3000/d/WM6WimE4z/crossings?orgId=1&refresh=5s

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

To create a custom Grafana dashboard, sign in with `admin\admin` credentials.

## Performance Measurement

Download the video file to your local folder. For example, create a data folder and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/leeds_1080p.mp4 https://eu-central-1.linodeobjects.com/savant-data/demo/leeds_1080p.mp4
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
./samples/intersection_traffic_meter/run_perf.sh
```
