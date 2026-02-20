# Retina RTSP RTCP SR Stream Synchronization

A pipeline demonstrating the use of the following features:

- Retina RTSP adapter;
- RTSP SR stream synchronization;
- Auxiliary streams.

Platforms:

- Nvidia Turing and newer.

Preview:

![](assets/cloud-sync.webp)


## Prerequisites

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant
git lfs pull
./utils/check-environment-compatible
```

**Note**: Ubuntu 22.04 runtime configuration [guide](https://insight-platform.github.io/savant-ai.io/docs/latest/getting_started/0_configure_prod_env.html) helps to configure the runtime to run Savant pipelines.

## Run Demo

### Download source video

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/clouds.mp4 https://eu-central-1.linodeobjects.com/savant-data/demo/clouds.mp4
```

### Run the demo


```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/retina_rtsp_rtcp_sr/docker-compose.yml up

# open 'rtsp://127.0.0.1:554/stream/composition' in your player
# or visit 'http://127.0.0.1:888/stream/composition' (LL-HLS)

# Ctrl+C to stop running the compose bundle
```

**Note**: When running this demo, you may receive a lot of warnings from MediaMTX like below:

```
2025/05/01 09:16:00 WAR [path overlay] [RTSP source] received RTP packet without absolute time, skipping it
```

This is OK, because MediaMTX is configured to wait for an RTCP SR before it accepts frames.