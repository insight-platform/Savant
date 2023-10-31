# Super Resolution

TBD

Preview:

![](assets/super_resolution_360p_1080p.webp)


Tested on platforms:
- Jetson Xavier NX/AGX;
- Nvidia Turing, Ampere.

## Run the demo

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/super_resolution

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
