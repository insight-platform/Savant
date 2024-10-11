# Pass-through Processing

A pipeline demonstrates how to pass frames from source to sink without content modifications and encoding. In the demo each part of the pipeline is running in a separate container. The pipeline consists of the following elements:

- detector;
- tracker;
- draw-func.

The modules performance is also stored in Prometheus and displayed on a Grafana dashboard.

![pass-through-demo.png](assets/pass-through-demo.png)

Detector and tracker are running in pass-through mode (`codec: copy`). Draw-func encodes frames to H264.

Tested on platforms:

- Nvidia Turing
- Nvidia Jetson Orin family

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

./scripts/run_module.py --build-engines samples/peoplenet_detector/module.yml
```

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/pass_through_processing/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/pass_through_processing/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/city-traffic' in your player
# or visit 'http://127.0.0.1:888/stream/city-traffic/' (LL-HLS)

# for pre-configured Grafana dashboard visit
# http://127.0.0.1:3000/d/a4c1f484-75c9-4375-a04d-ab5d50578239/module-performance-metrics?orgId=1&refresh=5s

# for the tracker metrics visit
# http://127.0.0.1:8000/metrics

# Ctrl+C to stop running the compose bundle
```

To create a custom Grafana dashboard, sign in with `admin\admin` credentials.
