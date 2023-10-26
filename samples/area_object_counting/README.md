# Area object counting demo

The pipeline determines if people are located in the delimited areas on the frame and counts the amount of people for each area displaying the counts on the sidebar.

Preview:

![](assets/area-object-counting-loop.webp)

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
cd Savant/samples/area_object_counting
git lfs pull

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

