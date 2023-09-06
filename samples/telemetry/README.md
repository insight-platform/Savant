# Telemetry Example

A simple pipeline demonstrating the possibilities of using OpenTelemetry with Savant. The pipeline contains only one element, [Blur PyFunc](blur.py). This element simply blurs the frame and contains sample code for working with telemetry. The pipeline [config](module.yml) does not contain any additional parameters. Savant has telemetry built in, and the user can enable it by simply specifying `jaeger` as the telemetry provider, and `entrypoint` using the `TELEMETRY_PROVIDER` and `TELEMETRY_PROVIDER_PARAMS` environment variables respectively. The entry point is a jaeger-agent from the [Jaeger All-in-One container](https://www.jaegertracing.io/docs/1.48/getting-started/#all-in-one). The container also includes the Jaeger UI.

Below are a few screenshots from the Jaeger UI.

#### Jaeger main screen
Select `telemetry-demo` **Service** and click **Find Traces** to see captured traces.

![Jaeger main screen](assets/00-main.png)

#### Trace view
Click any track to see the track timeline (the default track view).

![Trace view](assets/01-trace.png)

#### process-frame span
`process-frame` is the parent span for blur code in the [Blur PyFunc](blur.py).

![process-frame span](assets/02-process-frame.png)

#### blur-filter span
`blur-filter` is a telemetry span for the blur function call.

![blur-filter span](assets/03-blur-filter.png)

#### error-code span
The `error-code` span demonstrates the ability to catch exceptions using telemetry.

![error-code span](assets/04-error-code.png)

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
