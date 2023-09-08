import time
from savant_rs import init_jaeger_tracer
from savant.client import JaegerLogProvider, JpegSource, SourceBuilder

# Initialize Jaeger tracer to send metrics and logs to Jaeger.
# Note: the Jaeger tracer also should be configured in the module.
init_jaeger_tracer('savant-client', 'jaeger:6831')

# Build the source
source = (
    SourceBuilder()
    .with_log_provider(JaegerLogProvider('http://jaeger:16686'))
    .with_socket('pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc')
    # Note: healthcheck port should be configured in the module.
    .with_module_health_check_url('http://module:8888/healthcheck')
    .build()
)

# Send a JPEG image from a file to the module
result = source(JpegSource('cam-1', '/data/dwayne_johnson_01.jpg'))
print(result.status)
time.sleep(1)  # Wait for the module to process the frame
result.logs().pretty_print()
