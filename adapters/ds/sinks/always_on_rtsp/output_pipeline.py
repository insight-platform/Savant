from adapters.ds.sinks.always_on_rtsp.config import Config
from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrame
from adapters.ds.sinks.always_on_rtsp.pipeline import add_elements
from savant.config.schema import PipelineElement
from savant.gstreamer import Gst
from savant.gstreamer.element_factory import GstElementFactory
from savant.utils.platform import is_aarch64


def build_output_pipeline(
    config: Config,
    last_frame: LastFrame,
    factory: GstElementFactory,
) -> Gst.Pipeline:
    pipeline: Gst.Pipeline = Gst.Pipeline.new('output-pipeline')

    encoder_properties = {
        'profile': config.encoder_profile,
        'bitrate': config.encoder_bitrate,
    }
    if not is_aarch64():
        # nvv4l2h264enc doesn't encode video properly for the RTSP stream on dGPU
        # https://forums.developer.nvidia.com/t/rtsp-stream-sent-by-rtspclientsink-doesnt-play-in-deepstream-6-2/244194
        encoder_properties['tuning-info-id'] = 'HighQualityPreset'
    elements = [
        PipelineElement(
            'filesrc',
            properties={
                'location': str(config.stub_file_location.absolute()),
            },
        ),
        PipelineElement('jpegparse'),
        PipelineElement('jpegdec'),
        PipelineElement('imagefreeze'),
        PipelineElement(
            'nvvideoconvert',
            properties=config.nvvideoconvert_properties,
        ),
        PipelineElement(
            'capsfilter',
            properties={
                'caps': f'video/x-raw(memory:NVMM), format=RGBA, framerate={config.framerate}'
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
        PipelineElement(
            'nvvideoconvert',
            properties=config.nvvideoconvert_properties,
        ),
        PipelineElement(
            'nvv4l2h264enc',
            properties=encoder_properties,
        ),
        PipelineElement(
            'h264parse',
            properties={
                'config-interval': -1,
            },
        ),
        PipelineElement(
            'fps_meter',
            properties=config.fps_meter_properties('Output'),
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

    add_elements(pipeline, elements, factory)

    return pipeline
