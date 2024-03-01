import os
from typing import List, Optional

from savant.gstreamer.codecs import CODEC_BY_NAME, Codec
from savant.utils.config import opt_config, strtobool
from savant.utils.zeromq import ReceiverSocketTypes

SUPPORTED_CODECS = {x.value.name for x in [Codec.H264, Codec.HEVC]}


class Config:
    def __init__(self):
        self.dev_mode = opt_config('DEV_MODE', False, strtobool)
        self.source_id: Optional[str] = opt_config('SOURCE_ID')
        self.source_ids: Optional[List[str]] = opt_config(
            'SOURCE_IDS', [], lambda x: x.split(',')
        )
        assert (
            self.source_id or self.source_ids
        ), 'Either "SOURCE_ID" or "SOURCE_IDS" must be set.'
        if not self.source_ids:
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

        codec_name = opt_config('CODEC', 'h264')
        assert codec_name in SUPPORTED_CODECS, f'Unsupported codec {codec_name}.'
        self.codec = CODEC_BY_NAME[codec_name]

        self.api_port = opt_config('API_PORT', 13000, int)
        assert 1 <= self.api_port <= 65535, 'Invalid value for "API_PORT".'
