import functools
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from savant_rs.pipeline2 import (
    VideoPipeline,
    VideoPipelineConfiguration,
    VideoPipelineStagePayloadType,
)

from adapters.ds.sinks.always_on_rtsp.utils import nvidia_runtime_is_available
from savant.gstreamer.codecs import CODEC_BY_NAME, Codec
from savant.utils.config import opt_config, strtobool
from savant.utils.zeromq import ReceiverSocketTypes

ENCODER_DEFAULT_PROFILES = {
    Codec.H264: 'High',
    Codec.HEVC: 'Main',
}

ENCODER_PROFILES = {
    Codec.H264: ['Baseline', 'Main', 'High'],
    Codec.HEVC: ['Main', 'Main10', 'FREXT'],
}

SUPPORTED_CODECS = {x.value.name for x in [Codec.H264, Codec.HEVC]}

IDR_PERIOD_FRAMES = 30
MAX_ALLOWED_RESOLUTION = (3840, 2152)
ENCODER_BITRATE = 4000000


class TransferMode(str, Enum):
    SCALE_TO_FIT = 'scale-to-fit'
    CROP_TO_FIT = 'crop-to-fit'


class MetadataOutput(str, Enum):
    LOGGER = 'logger'
    STDOUT = 'stdout'


class CommonStreamConfig:
    def __init__(self):
        self.stub_file_location = Path(os.environ['STUB_FILE_LOCATION'])
        if not self.stub_file_location.exists():
            raise RuntimeError(f'File {self.stub_file_location} does not exist.')
        if not self.stub_file_location.is_file():
            raise RuntimeError(f'{self.stub_file_location} is not a file.')

        self.max_delay_ms = opt_config('MAX_DELAY_MS', 1000, int)
        try:
            self.transfer_mode: TransferMode = opt_config(
                'TRANSFER_MODE',
                TransferMode.SCALE_TO_FIT,
                TransferMode,
            )
        except ValueError:
            raise ValueError('Invalid value for environment variable TRANSFER_MODE')

        self.rtsp_protocols = opt_config('RTSP_PROTOCOLS', 'tcp')
        self.rtsp_latency_ms = opt_config('RTSP_LATENCY_MS', 100, int)
        self.rtsp_keep_alive = opt_config('RTSP_KEEP_ALIVE', True, strtobool)

        codec_name = opt_config('CODEC', 'h264')
        assert codec_name in SUPPORTED_CODECS, f'Unsupported codec {codec_name}.'
        self.codec = CODEC_BY_NAME[codec_name]
        self.encoder_profile = opt_config(
            'ENCODER_PROFILE', ENCODER_DEFAULT_PROFILES[self.codec]
        )
        assert self.encoder_profile in ENCODER_PROFILES[self.codec], (
            f'Invalid value for environment variable ENCODER_PROFILE. '
            f'Available profiles for {self.codec.value.name} are: '
            f'{", ".join(ENCODER_PROFILES[self.codec])}'
        )
        # default encoding bitrate
        self.encoder_bitrate = opt_config('ENCODER_BITRATE', ENCODER_BITRATE, int)

        self.fps_period_frames = opt_config('FPS_PERIOD_FRAMES', 1000, int)
        self.fps_period_seconds = opt_config('FPS_PERIOD_SECONDS', convert=float)
        self.fps_output = opt_config('FPS_OUTPUT', 'stdout')

        try:
            self.metadata_output = opt_config('METADATA_OUTPUT', None, MetadataOutput)
        except ValueError:
            raise ValueError('Invalid value for environment variable METADATA_OUTPUT')

        self.framerate = opt_config('FRAMERATE', '30/1')
        self.idr_period_frames = opt_config('IDR_PERIOD_FRAMES', IDR_PERIOD_FRAMES, int)

        self.sync = opt_config('SYNC_OUTPUT', False, strtobool)
        self.max_allowed_resolution = opt_config(
            'MAX_RESOLUTION',
            MAX_ALLOWED_RESOLUTION,
            lambda x: tuple(map(int, x.split('x'))),
        )

        assert len(self.max_allowed_resolution) == 2, (
            'Incorrect value for environment variable MAX_RESOLUTION, '
            'you should specify the width and height of the maximum resolution '
            'in format WIDTHxHEIGHT, for example 1920x1080.'
        )


class Config(CommonStreamConfig):
    def __init__(self):
        super().__init__()

        self.source_id = os.environ['SOURCE_ID']

        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.zmq_socket_type = opt_config(
            'ZMQ_TYPE',
            ReceiverSocketTypes.SUB,
            ReceiverSocketTypes.__getitem__,
        )
        self.zmq_socket_bind = opt_config('ZMQ_BIND', False, strtobool)

        self.rtsp_uri = os.environ['RTSP_URI']

        self.pipeline_source_stage_name = 'source'
        self.pipeline_demux_stage_name = 'source-demux'
        self.video_pipeline: Optional[VideoPipeline] = VideoPipeline(
            'always-on-sink',
            [
                (self.pipeline_source_stage_name, VideoPipelineStagePayloadType.Frame),
                (self.pipeline_demux_stage_name, VideoPipelineStagePayloadType.Frame),
            ],
            VideoPipelineConfiguration(),
        )

    @functools.cached_property
    def converter(self) -> str:
        return 'nvvideoconvert' if nvidia_runtime_is_available() else 'videoconvert'

    @functools.cached_property
    def video_raw_caps(self) -> str:
        return (
            'video/x-raw(memory:NVMM)'
            if nvidia_runtime_is_available()
            else 'video/x-raw'
        )

    def fps_meter_properties(self, measurer_name: str):
        props = {'output': self.fps_output, 'measurer-name': measurer_name}
        if self.fps_period_seconds:
            props['period-seconds'] = self.fps_period_seconds
        else:
            props['period-frames'] = self.fps_period_frames
        return props
