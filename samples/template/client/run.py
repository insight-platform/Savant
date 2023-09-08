import json
import time

import cv2
import numpy as np
from savant_rs import init_jaeger_tracer

from savant.api.builder import build_bbox
from savant.client import JaegerLogProvider, JpegSource, SinkBuilder, SourceBuilder

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

# Specify a JPEG image to send to the module
src_jpeg = JpegSource(source_id, '/test_data/test_img.jpg')
with open('/test_data/test_img.json', 'r', encoding='utf8') as f:
    src_meta = json.loads(f.read())

# the test image has 1 person and 1 face
person_bbox, face_bbox = None, None
for obj in src_meta['objects']:
    if obj['label'] == 'person':
        person_bbox = build_bbox(obj['bbox'])
    elif obj['label'] == 'face':
        face_bbox = build_bbox(obj['bbox'])

# Send a JPEG image from a file to the module
# And then send an EOS message
result = source(src_jpeg, send_eos=True)

time.sleep(1)  # Wait for the module to process the frame

# Receive results from the module and print them
for result in sink:
    if result.eos:
        # second message is the EOS
        print('EOS')
        # Optionally send a shutdown message to the module
        # source.send_shutdown(source_id, shutdown_auth)
        break

    # first message is the module result for the sent JPEG
    # check that the result is correct
    # for simplicity, we only check the IOU coefs of bounding boxes
    for obj in result.frame_meta.get_all_objects():
        if obj.label == 'person':
            assert (
                obj.detection_box.iou(person_bbox) > 0.9
            ), 'Person bbox is not correct'
        elif obj.label == 'face':
            assert obj.detection_box.iou(face_bbox) > 0.9, 'Face bbox is not correct'

    # get the result image
    # the image will be in RGBA format, as specified in the module config
    img = np.frombuffer(result.frame_content, dtype=np.uint8)
    img = img.reshape(result.frame_meta.height, result.frame_meta.width, 4)

    # save the result image
    # the image will anything that the module has drawn on top of the input image
    cv2.imwrite('/output/result_img.jpg', cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))

    # print the processing logs from the module
    result.logs().pretty_print()
