# Source Shaper Sample

A pipeline demonstrating the use of custom source shaper to resize and add padding to the input video streams independently. The sample accepts two video stream with different resolutions and reshapes them according to the configuration in [module.yml](module.yml). Paddings are colored for demonstration purposes.

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
docker compose -f samples/source_shaper_sample/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/source_shaper_sample/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/city-traffic', 'rtsp://127.0.0.1:554/stream/town-centre' in your player
# or visit 'http://127.0.0.1:888/stream/city-traffic', 'http://127.0.0.1:888/stream/town-centre' (LL-HLS)

# All the streams on AO-sink are 1280x720 for simplicity of deployment.
# Paddings added by the module are colored.

# Ctrl+C to stop running the compose bundle
```
