from adapters.ds.sinks.always_on_rtsp.config import Config
from adapters.ds.sinks.always_on_rtsp.encoder_builder import build_encoder_elements
from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrameRef
from adapters.ds.sinks.always_on_rtsp.pipeline import add_elements
from savant.config.schema import PipelineElement
from savant.gstreamer import Gst
from savant.gstreamer.element_factory import GstElementFactory


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
                    'mode': config.transfer_mode.value,
                    'last-frame': last_frame,
                },
            ),
            PipelineElement(config.converter),
        ]
        + build_encoder_elements(config)
        + [
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
