# OpenTelemetry Example

A simple pipeline demonstrating the use of OpenTelemetry in Savant. The pipeline contains one element, [Blur PyFunc](blur.py). It applies gaussian blur to the frame and contains OpenTelemetry instrumenting code. OpenTelemetry instrumenting is initialized with environment variables: see compose files for details. The telemetry is collected by the all-in-one Jaeger [container](https://www.jaegertracing.io/docs/1.48/getting-started/#all-in-one). The container also includes the Jaeger UI.

Below are a few screenshots from the Jaeger UI.

#### Jaeger main screen

Select `telemetry-demo` **Service** and click **Find Traces** to see captured traces.

![Jaeger main screen](assets/00-main.png)

#### Trace View

Click on a trace to see the trace spans, associated logging messages, events and attributes.

![Trace view](assets/01-trace.png)

#### Process-Frame Span

The `process-frame` span is the parent span for blurring code located in the [Blur PyFunc](blur.py).

![process-frame span](assets/02-process-frame.png)

#### Blur-Filter Span

The `blur-filter` span is a telemetry span for the blur function call.
 
![blur-filter span](assets/03-blur-filter.png)

#### Error-Code Span

The `error-code` span demonstrates the span capability of attaching uncaught exceptions to the span.

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
