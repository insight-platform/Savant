import os
from typing import List, Optional

from adapters.ds.sinks.always_on_rtsp.config import CommonStreamConfig
from savant.utils.config import opt_config, strtobool
from savant.utils.zeromq import ReceiverSocketTypes


class AppConfig(CommonStreamConfig):
    def __init__(self):
        super().__init__()

        self.dev_mode = opt_config('DEV_MODE', False, strtobool)
        self.source_id: Optional[str] = opt_config('SOURCE_ID')
        self.source_ids: Optional[List[str]] = opt_config(
            'SOURCE_IDS', [], lambda x: x.split(',')
        )
        if self.source_id and not self.source_ids:
            self.source_ids = [self.source_id]

        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.zmq_socket_type = opt_config(
            'ZMQ_TYPE',
            ReceiverSocketTypes.SUB,
            ReceiverSocketTypes.__getitem__,
        )
        self.zmq_socket_bind = opt_config('ZMQ_BIND', False, strtobool)

        self.rtsp_uri = opt_config('RTSP_URI')
        if self.dev_mode:
            assert (
                self.rtsp_uri is None
            ), '"RTSP_URI" cannot be set when "DEV_MODE=True"'
            self.rtsp_uri = 'rtsp://localhost:554/stream'
        else:
            assert (
                self.rtsp_uri is not None
            ), '"RTSP_URI" must be set when "DEV_MODE=False"'

        self.api_port = opt_config('API_PORT', 13000, int)
        assert 1 <= self.api_port <= 65535, 'Invalid value for "API_PORT".'

        self.status_poll_interval_ms = opt_config('STATUS_POLL_INTERVAL_MS', 1000, int)
        assert (
            self.status_poll_interval_ms > 0
        ), 'Invalid value for "STATUS_POLL_INTERVAL_MS".'
        self.status_poll_interval = self.status_poll_interval_ms / 1000

        self.fail_on_stream_error = opt_config('FAIL_ON_STREAM_ERROR', True, strtobool)
