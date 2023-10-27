# Keypoint detection demo

This application demonstrates the use of the human body key point detection model.

Preview:

![](assets/shuffle_dance.webp)

Tested on platforms:

- Xavier NX, Xavier AGX, Orin Nano;
- Nvidia Turing, Ampere.

Demonstrated adapters:

- Video loop source adapter;
- Always-ON RTSP sink adapter.

**Note**: Ubuntu 22.04 runtime configuration [guide](../../docs/runtime-configuration.md) helps to configure the runtime to run Savant pipelines.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/keypoint_detection

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```

## Performance Measurement

Download the video file to your local folder. For example, create a data folder and download the video into it (all commands must be executed from the root directory of the project Savant)

```bash
# you are expected to be in Savant/ directory

mkdir -p data && curl -o data/lpr_test_1080p.mp4 \
   https://eu-central-1.linodeobjects.com/savant-data/demo/shuffle_dance.mp4
```

Now you are ready to run the performance benchmark with the following command:

```bash
./samples/keypoint_detection/run_perf.sh
```

