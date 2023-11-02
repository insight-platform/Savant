from typing import List

from adapters.ds.sinks.always_on_rtsp.config import Config
from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrameRef
from adapters.ds.sinks.always_on_rtsp.pipeline import add_elements
from savant.config.schema import PipelineElement
from savant.gstreamer import Gst
from savant.gstreamer.element_factory import GstElementFactory
from savant.utils.platform import is_aarch64


def build_output_pipeline(
    config: Config,
    last_frame: LastFrameRef,
    factory: GstElementFactory,
) -> Gst.Pipeline:
    pipeline: Gst.Pipeline = Gst.Pipeline.new('output-pipeline')

    elements = (
        [
            PipelineElement(
                'filesrc',
                properties={
                    'location': str(config.stub_file_location.absolute()),
                },
            ),
            PipelineElement('jpegparse'),
            PipelineElement('jpegdec'),
            PipelineElement('imagefreeze'),
            PipelineElement(config.converter),
            PipelineElement(
                'capsfilter',
                properties={
                    'caps': f'{config.video_raw_caps}, format=RGBA, framerate={config.framerate}'
                },
            ),
            PipelineElement(
                'always_on_rtsp_frame_processor',
                properties={
                    'max-delay-ms': config.max_delay_ms,
                    'mode': config.transfer_mode,
                    'last-frame': last_frame,
                },
            ),
            PipelineElement(config.converter),
        ]
        + build_encoder_elements(config)
        + [
            PipelineElement(
                'h264parse',
                properties={
                    'config-interval': -1,
                },
            ),
            PipelineElement(
                'fps_meter',
                properties=config.fps_meter_properties(f'Output {config.source_id}'),
            ),
            PipelineElement(
                'rtspclientsink',
                properties={
                    'location': config.rtsp_uri,
                    'protocols': config.rtsp_protocols,
                    'latency': config.rtsp_latency_ms,
                    'do-rtsp-keep-alive': config.rtsp_keep_alive,
                },
            ),
        ]
    )

    add_elements(pipeline, elements, factory)

    return pipeline


def build_encoder_elements(config: Config) -> List[PipelineElement]:
    if config.nvidia_runtime_is_available:
        return build_nvenc_encoder_elements(config)
    else:
        return build_sw_encoder_elements(config)


def build_nvenc_encoder_elements(config: Config) -> List[PipelineElement]:
    properties = {
        'profile': config.encoder_profile,
        'bitrate': config.encoder_bitrate,
    }
    if not is_aarch64():
        # nvv4l2h264enc doesn't encode video properly for the RTSP stream on dGPU
        # https://forums.developer.nvidia.com/t/rtsp-stream-sent-by-rtspclientsink-doesnt-play-in-deepstream-6-2/244194
        properties['tuning-info-id'] = 'HighQualityPreset'

    return [PipelineElement('nvv4l2h264enc', properties=properties)]


def build_sw_encoder_elements(config: Config) -> List[PipelineElement]:
    return [
        PipelineElement(
            'x264enc',
            properties={
                'tune': 'zerolatency',
                'bitrate': config.encoder_bitrate // 1024,  # bit/s -> kbit/s
                'speed-preset': 'veryfast',
            },
        ),
        PipelineElement(
            'capsfilter',
            properties={
                'caps': f'video/x-h264,profile={config.encoder_profile.lower()}'
            },
        ),
    ]
