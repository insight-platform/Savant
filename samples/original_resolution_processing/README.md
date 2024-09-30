# Original Resolution Processing

A pipeline demonstrates processing of streams at the original resolution, i.e. without scaling to a single resolution. `parameters.frame` in  [module.yml](module.yml) is not specified. The sample sends two streams with resolutions 1280x720 and 1920x1080 to the module.

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
docker compose -f samples/different_resolutions/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/different_resolutions/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/video-720p' and 'rtsp://127.0.0.1:554/stream/video-1080p' in your player
# or visit 'http://127.0.0.1:888/stream/video-720p/' and 'http://127.0.0.1:888/stream/video-720p/' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

**Note**: All the streams on AO-sink are 1280x720 for simplicity of deployment. You can see the different quality of the streams since AO-sink upscales them to 1920x1080. Additionally, you can see the resolution of the streams in the logs of ao-sink:

```
always-on-sink-1 |  INFO  insight::savant::savant_rs_video_demux     > Created new src pad for source video-720p: src_video-720p.
always-on-sink-1 |  INFO  insight::savant::always_on_rtsp_frame_sink > Frame resolution is 1280x720
...
always-on-sink-1 |  INFO  insight::savant::savant_rs_video_demux     > Created new src pad for source video-1080p: src_video-1080p.
always-on-sink-1 |  INFO  insight::savant::always_on_rtsp_frame_sink > Frame resolution is 1920x1080
```
