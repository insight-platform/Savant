# Buffer Adapter Demo

A pipeline demonstrates how Buffer Adapter works in Savant. In the demo video from Video Loop Source adapter passed to Buffer Adapter. Buffer Adapter stores frames in buffer and passes it to the module. Then the module passes the processed frames to Always-On-RTSP Sink adapter. The module simulates a periodic load spike by adding a lag for 0.06 - 0.1 seconds to 500 frames after every 500 frames without a lag. When there is a load spike, the buffer adapter stores frames in the buffer and passes them to the module when the load is reduced. When the buffer is full, the buffer adapter drops incoming frames.

The buffer adapter metrics are stored in Prometheus and displayed on a Grafana dashboard.

## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime configuration [guide](https://insight-platform.github.io/Savant/develop/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

## Run Demo

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/buffer_adapter/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/buffer_adapter/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/city-traffic' in your player
# or visit 'http://127.0.0.1:888/stream/city-traffic/' (LL-HLS)

# for pre-configured Grafana dashboard visit
# http://127.0.0.1:3000/d/89571523-ad22-4df2-bb09-df20b18bd5ee/buffer-metrics?orgId=1&refresh=5s

# for the buffer adapter metrics visit
# http://127.0.0.1:8000/metrics


# Ctrl+C to stop running the compose bundle
```
