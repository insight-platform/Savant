import json
import os
import time

import cv2
import numpy as np
from savant_rs import telemetry
from savant_rs.telemetry import (
    ContextPropagationFormat,
    Protocol,
    TelemetryConfiguration,
    TracerConfiguration,
)

from savant.api.builder import build_bbox
from savant.client import JaegerLogProvider, JpegSource, SinkBuilder, SourceBuilder

module_hostname = os.environ.get('MODULE_HOSTNAME', 'localhost')
healthcheck_url = f'http://{module_hostname}:8888/status'
source_id = 'test-source'
parent_dir = os.path.dirname(os.path.dirname(__file__))
result_img_path = os.path.join(parent_dir, 'output', 'result_img.jpeg')

# Build the source
source = (
    SourceBuilder()
    .with_socket('dealer+connect:ipc:///tmp/zmq-sockets/input-video.ipc')
    .with_module_health_check_url(healthcheck_url)
    .build()
)

counter = 0
jpeg1 = JpegSource(source_id, file='/test_data/test_img.jpeg')
while True:
    _ = source(jpeg1, send_eos=True)
    counter += 1
    print(f'Counter: {counter}')
