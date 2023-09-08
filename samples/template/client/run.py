import time
import json
import numpy as np
import cv2

from savant_rs import init_jaeger_tracer
from savant.api.builder import build_bbox
from savant.client import JaegerLogProvider, JpegSource, SourceBuilder, SinkBuilder

# Initialize Jaeger tracer to send metrics and logs to Jaeger.
# Note: the Jaeger tracer also should be configured in the module.
init_jaeger_tracer('savant-client', 'jaeger:6831')

jaeger_endpoint = 'http://jaeger:16686'
healthcheck_url = 'http://module:8888/healthcheck'
source_id = 'test-source'
shutdown_auth = 'shutdown'

# Build the source
source = (
    SourceBuilder()
    .with_log_provider(JaegerLogProvider(jaeger_endpoint))
    .with_socket('pub+connect:ipc:///tmp/zmq-sockets/input-video.ipc')
    # Note: healthcheck port should be configured in the module.
    .with_module_health_check_url(healthcheck_url)
    .build()
)

sink = (
    SinkBuilder()
    .with_socket('sub+connect:ipc:///tmp/zmq-sockets/output-video.ipc')
    .with_idle_timeout(60)
    .with_log_provider(JaegerLogProvider(jaeger_endpoint))
    # Note: healthcheck port should be configured in the module.
    .with_module_health_check_url(healthcheck_url)
    .build()
)

src_jpeg = JpegSource(source_id, '/data/test_img.jpg')
with open('/data/test_img.json', 'r') as f:
    src_meta = json.loads(f.read())

person_bbox, face_bbox = None, None
for obj in src_meta['objects']:
    if obj['label'] == 'person':
        person_bbox = build_bbox(obj['bbox'])
    elif obj['label'] == 'face':
        face_bbox = build_bbox(obj['bbox'])

# Send a JPEG image from a file to the module
result = source(src_jpeg)

time.sleep(1)  # Wait for the module to process the frame

# Receive results from the module and print them
for result in sink:
    if result.eos:
        print('EOS')
        # source.send_shutdown(source_id, shutdown_auth)
        break

    for obj in result.frame_meta.get_all_objects():
        if obj.label == 'person':
            assert  obj.detection_box.iou(person_bbox) > 0.9, 'Person bbox is not correct'
        elif obj.label == 'face':
            assert obj.detection_box.iou(face_bbox) > 0.9, 'Face bbox is not correct'

    img = np.frombuffer(result.frame_content, dtype=np.uint8)
    img = img.reshape(result.frame_meta.height, result.frame_meta.width, 4)

    cv2.imwrite('/data/result_img.jpg', cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    result.logs().pretty_print()
