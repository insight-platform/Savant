# Pass-through Processing

A pipeline demonstrates how to pass frames from source to sink without content modifications and encoding. In the demo each part of the pipeline is running in a separate container. The pipeline consists of the following elements:

- detector;
- tracker;
- draw-func.

The modules performance is also stored in Prometheus and displayed on a Grafana dashboard.

![pass-through-demo.png](assets/pass-through-demo.png)

Detector and tracker are running in pass-through mode (`codec: copy`). Draw-func encodes frames to H264.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/pass_through_processing
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# for pre-configured Grafana dashboard visit
# http://127.0.0.1:3000/d/a4c1f484-75c9-4375-a04d-ab5d50578239/module-performance-metrics?orgId=1&refresh=5s

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

To create a custom Grafana dashboard, sign in with `admin\admin` credentials.
