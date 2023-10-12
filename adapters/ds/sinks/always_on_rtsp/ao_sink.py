import time
from datetime import datetime

from adapters.ds.sinks.always_on_rtsp.config import Config
from adapters.ds.sinks.always_on_rtsp.input_pipeline import build_input_pipeline
from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrame
from adapters.ds.sinks.always_on_rtsp.output_pipeline import build_output_pipeline
from adapters.ds.sinks.always_on_rtsp.pipeline import PipelineThread
from savant.deepstream.element_factory import NvDsElementFactory
from savant.gstreamer import Gst
from savant.utils.logging import get_logger, init_logging

LOGGER_NAME = 'adapters.ao_sink'
logger = get_logger(LOGGER_NAME)


def run_ao_sink():
    init_logging()
    config = Config()

    last_frame = LastFrame(timestamp=datetime.utcfromtimestamp(0))

    Gst.init(None)

    logger.info('Source %s. Starting Always-On-RTSP sink', config.source_id)
    factory = NvDsElementFactory()
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
