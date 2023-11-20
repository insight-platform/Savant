import time

from adapters.ds.sinks.always_on_rtsp.config import Config
from adapters.ds.sinks.always_on_rtsp.input_pipeline import build_input_pipeline
from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrameRef
from adapters.ds.sinks.always_on_rtsp.output_pipeline import build_output_pipeline
from adapters.ds.sinks.always_on_rtsp.pipeline import PipelineThread
from adapters.ds.sinks.always_on_rtsp.utils import nvidia_runtime_is_available
from savant.gstreamer import Gst
from savant.gstreamer.element_factory import GstElementFactory
from savant.utils.logging import get_logger, init_logging

LOGGER_NAME = 'adapters.ao_sink'
logger = get_logger(LOGGER_NAME)


def create_element_factory() -> GstElementFactory:
    if nvidia_runtime_is_available():
        from savant.deepstream.element_factory import NvDsElementFactory

        logger.debug('Using NvDsElementFactory')
        return NvDsElementFactory()

    logger.debug('Using GstElementFactory')
    return GstElementFactory()


def run_ao_sink():
    init_logging()
    config = Config()

    last_frame = LastFrameRef()

    Gst.init(None)

    logger.info('Source %s. Starting Always-On-RTSP sink', config.source_id)
    factory = create_element_factory()
    output_pipeline_thread = PipelineThread(
        build_output_pipeline,
        'OutputPipeline',
        config,
        last_frame,
        factory,
    )
    input_pipeline_thread = PipelineThread(
        build_input_pipeline,
        'InputPipeline',
        config,
        last_frame,
        factory,
    )
    output_pipeline_thread.start()
    try:
        ao_sink_loop(output_pipeline_thread, input_pipeline_thread)
    except KeyboardInterrupt:
        pass
    logger.info('Source %s. Stopping Always-On-RTSP sink', config.source_id)
    input_pipeline_thread.stop()
    output_pipeline_thread.stop()
    logger.info('Source %s. Always-On-RTSP sink stopped', config.source_id)


def ao_sink_loop(
    output_pipeline_thread: PipelineThread,
    input_pipeline_thread: PipelineThread,
):
    while output_pipeline_thread.is_running:
        input_pipeline_thread.start()
        while input_pipeline_thread.is_running and output_pipeline_thread.is_running:
            time.sleep(1)


if __name__ == '__main__':
    run_ao_sink()
