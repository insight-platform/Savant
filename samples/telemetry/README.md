# OpenTelemetry Example

A simple pipeline demonstrating the use of OpenTelemetry in Savant. The pipeline contains one element, [Blur PyFunc](blur.py). It applies gaussian blur to the frame and contains OpenTelemetry instrumenting code. OpenTelemetry instrumenting is initialized with environment variables: see compose files for details. The telemetry is collected by the all-in-one Jaeger [container](https://www.jaegertracing.io/docs/1.62/getting-started/#all-in-one). The container also includes the Jaeger UI.

Below are a few screenshots from the Jaeger UI.

## Jaeger Main Screen

Select `telemetry-demo` **Service** and click **Find Traces** to see captured traces.

![Jaeger main screen](assets/00-main.png)

## Trace View

Click on a trace to see the trace spans, associated logging messages, events and attributes.

![Trace view](assets/01-trace.png)

## Process-Frame Span

The `process-frame` span is the parent span for blurring code located in the [Blur PyFunc](blur.py).

![process-frame span](assets/02-process-frame.png)

## Blur-Filter Span

The `blur-filter` span is a telemetry span for the blur function call.
 
![blur-filter span](assets/03-blur-filter.png)

## Error-Code Span

The `error-code` span demonstrates the span capability of attaching uncaught exceptions to the span.

![error-code span](assets/04-error-code.png)

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
docker compose -f samples/telemetry/docker-compose.x86.yml up

# if Jetson
docker compose -f samples/telemetry/docker-compose.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/0' in your player
# or visit 'http://127.0.0.1:888/stream/0/' (LL-HLS)

# navigate to 'http://localhost:16686' to access the Jaeger UI

# Ctrl+C to stop running the compose bundle
```

## Running Jaeger with TLS

Jaeger and Savant support TLS for telemetry collection. [docker-compose-with-tls.x86.yml](docker-compose-with-tls.x86.yml) and [docker-compose-with-tls.l4t.yml](docker-compose-with-tls.l4t.yml) compose files demonstrate how to run Jaeger with TLS enabled (for both Jaeger and Savant authentication).

### Generate Certificates

Create directory for certificates:

```bash
mkdir samples/telemetry/certs
```

Generate CA:

```bash
openssl genpkey -algorithm RSA -out samples/telemetry/certs/ca.key
openssl req -new -x509 -days 365 \
    -key samples/telemetry/certs/ca.key \
    -out samples/telemetry/certs/ca.crt \
    -subj "/CN=jaeger-ca"
```

Generate server key and certificate:

```bash
openssl genpkey -algorithm RSA -out samples/telemetry/certs/server.key
openssl req -new \
    -key samples/telemetry/certs/server.key \
    -out samples/telemetry/certs/server.csr \
    -subj "/CN=jaeger"
openssl x509 -req -days 365 \
    -in samples/telemetry/certs/server.csr \
    -CA samples/telemetry/certs/ca.crt \
    -CAkey samples/telemetry/certs/ca.key \
    -CAcreateserial \
    -out samples/telemetry/certs/server.crt \
    -extfile <(echo "subjectAltName=DNS:jaeger")
```

Make the server key readable by the container:

```bash
chmod +r samples/telemetry/certs/server.key
```

Generate client key and certificate:

```bash
openssl genpkey -algorithm RSA -out samples/telemetry/certs/client.key
openssl req -new \
    -key samples/telemetry/certs/client.key \
    -out samples/telemetry/certs/client.csr \
    -subj "/CN=module"
openssl x509 -req -days 365 \
    -in samples/telemetry/certs/client.csr \
    -CA samples/telemetry/certs/ca.crt \
    -CAkey samples/telemetry/certs/ca.key \
    -CAcreateserial \
    -out samples/telemetry/certs/client.crt \
    -extfile <(echo "subjectAltName=DNS:module")
```

### Run Demo with TLS

```bash
# you are expected to be in Savant/ directory

# if x86
docker compose -f samples/telemetry/docker-compose-with-tls.x86.yml up

# if Jetson
docker compose -f samples/telemetry/docker-compose-with-tls.l4t.yml up

# open 'rtsp://127.0.0.1:554/stream/0' in your player
# or visit 'http://127.0.0.1:888/stream/0/' (LL-HLS)

# navigate to 'http://localhost:16686' to access the Jaeger UI

# Ctrl+C to stop running the compose bundle
```
