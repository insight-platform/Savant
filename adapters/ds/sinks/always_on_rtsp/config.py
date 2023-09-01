import os
from distutils.util import strtobool
from pathlib import Path
from typing import Optional

import pyds
from savant_rs.pipeline2 import VideoPipeline, VideoPipelineStagePayloadType

from savant.utils.platform import is_aarch64
from savant.utils.zeromq import ReceiverSocketTypes


def opt_config(name, default=None, convert=None):
    conf_str = os.environ.get(name)
    if conf_str:
        return convert(conf_str) if convert else conf_str
    return default


class Config:
    def __init__(self):
        self.stub_file_location = Path(os.environ['STUB_FILE_LOCATION'])
        if not self.stub_file_location.exists():
            raise RuntimeError(f'File {self.stub_file_location} does not exist.')
        if not self.stub_file_location.is_file():
            raise RuntimeError(f'{self.stub_file_location} is not a file.')

        self.max_delay_ms = opt_config('MAX_DELAY_MS', 1000, int)
        self.transfer_mode = opt_config('TRANSFER_MODE', 'scale-to-fit')
        self.source_id = os.environ['SOURCE_ID']

        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.zmq_socket_type = opt_config(
            'ZMQ_TYPE',
            ReceiverSocketTypes.SUB,
            ReceiverSocketTypes.__getitem__,
        )
        self.zmq_socket_bind = opt_config('ZMQ_BIND', False, strtobool)

        self.rtsp_uri = os.environ['RTSP_URI']
        self.rtsp_protocols = opt_config('RTSP_PROTOCOLS', 'tcp')
        self.rtsp_latency_ms = opt_config('RTSP_LATENCY_MS', 100, int)
        self.rtsp_keep_alive = opt_config('RTSP_KEEP_ALIVE', True, strtobool)

        self.encoder_profile = opt_config('ENCODER_PROFILE', 'High')
        # default nvv4l2h264enc bitrate
        self.encoder_bitrate = opt_config('ENCODER_BITRATE', 4000000, int)

        self.fps_period_frames = opt_config('FPS_PERIOD_FRAMES', 1000, int)
        self.fps_period_seconds = opt_config('FPS_PERIOD_SECONDS', convert=float)
        self.fps_output = opt_config('FPS_OUTPUT', 'stdout')

        self.metadata_output = opt_config('METADATA_OUTPUT')
        if self.metadata_output:
            self.pipeline_stage_name = 'source'
            self.video_pipeline: Optional[VideoPipeline] = VideoPipeline(
                'always-on-sink',
                [(self.pipeline_stage_name, VideoPipelineStagePayloadType.Frame)],
            )
        else:
            self.pipeline_stage_name = None
            self.video_pipeline: Optional[VideoPipeline] = None

        self.framerate = opt_config('FRAMERATE', '30/1')
        self.sync = opt_config('SYNC_OUTPUT', False, strtobool)

    def fps_meter_properties(self, measurer_name: str):
        props = {'output': self.fps_output, 'measurer-name': measurer_name}
        if self.fps_period_seconds:
            props['period-seconds'] = self.fps_period_seconds
        else:
            props['period-frames'] = self.fps_period_frames
        return props

    @property
    def nvvideoconvert_properties(self):
        props = {}
        if not is_aarch64():
            props['nvbuf-memory-type'] = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        return props
