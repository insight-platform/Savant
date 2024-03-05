from abc import ABC, abstractmethod
from typing import List

from adapters.ds.sinks.always_on_rtsp.config import Config
from adapters.ds.sinks.always_on_rtsp.utils import nvidia_runtime_is_available
from savant.config.schema import PipelineElement
from savant.gstreamer.codecs import Codec


class BaseEncoderBuilder(ABC):
    """Base builder for encoder elements."""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def build_encoder_elements(self) -> List[PipelineElement]:
        """Builds elements for encoder."""

    @abstractmethod
    def build_parser_elements(self) -> List[PipelineElement]:
        """Builds elements for parser."""


class H264EncoderBuilder(BaseEncoderBuilder):
    """Builder for H264 encoder elements."""

    def build_encoder_elements(self) -> List[PipelineElement]:
        if nvidia_runtime_is_available():
            return self.build_nvenc_encoder_elements()
        else:
            return self.build_sw_encoder_elements()

    def build_nvenc_encoder_elements(self) -> List[PipelineElement]:
        properties = {
            'profile': self.config.encoder_profile,
            'bitrate': self.config.encoder_bitrate,
            'iframeinterval': self.config.idr_period_frames,
        }

        return [PipelineElement('nvv4l2h264enc', properties=properties)]

    def build_sw_encoder_elements(self) -> List[PipelineElement]:
        return [
            PipelineElement(
                'x264enc',
                properties={
                    'tune': 'zerolatency',
                    'bitrate': self.config.encoder_bitrate // 1024,  # bit/s -> kbit/s
                    'speed-preset': 'veryfast',
                    'key-int-max': self.config.idr_period_frames,
                },
            ),
            PipelineElement(
                'capsfilter',
                properties={
                    'caps': f'video/x-h264,profile={self.config.encoder_profile.lower()}'
                },
            ),
        ]

    def build_parser_elements(self) -> List[PipelineElement]:
        return [PipelineElement('h264parse', properties={'config-interval': -1})]


class HevcEncoderBuilder(BaseEncoderBuilder):
    """Builder for HEVC encoder elements."""

    def build_encoder_elements(self) -> List[PipelineElement]:
        properties = {
            'profile': self.config.encoder_profile,
            'bitrate': self.config.encoder_bitrate,
            'iframeinterval': self.config.idr_period_frames,
        }

        return [PipelineElement('nvv4l2h265enc', properties=properties)]

    def build_parser_elements(self) -> List[PipelineElement]:
        return [PipelineElement('h265parse', properties={'config-interval': -1})]


ENCODER_BUILDER_CLASS = {
    Codec.H264: H264EncoderBuilder,
    Codec.HEVC: HevcEncoderBuilder,
}


def build_encoder_elements(config: Config) -> List[PipelineElement]:
    try:
        builder = ENCODER_BUILDER_CLASS[config.codec](config)
    except KeyError:
        raise ValueError(f'Unsupported encoder: {config.codec}')

    elements = builder.build_encoder_elements() + builder.build_parser_elements()

    return elements
