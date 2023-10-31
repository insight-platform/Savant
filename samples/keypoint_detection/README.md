# Keypoint detection demo

This application demonstrates the human body key point detection model.

Preview:

![](assets/shuffle_dance.webp)

Tested on platforms:

- Xavier NX, Xavier AGX;
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

