# Telemetry Example

TODO: Describe what the sample does and how the telemetry is configured. 

A simple pipeline demonstrating the possibilities of using OpenTelemetry with Savant.

Preview:

TODO: Add a screenshot of a trace from Jaeger.

Run the demo:

```bash
git clone https://github.com/insight-platform/Savant.git
cd Savant/samples/telemetry
git lfs pull

# if x86
../../utils/check-environment-compatible && docker compose -f docker-compose.x86.yml up

# if Jetson
../../utils/check-environment-compatible && docker compose -f docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream' in your player
# or visit 'http://127.0.0.1:888/stream/' (LL-HLS)

# navigate to 'http://localhost:16686' to access the Jaeger UI

# Ctrl+C to stop running the compose bundle

# to get back to project root
cd ../..
```
