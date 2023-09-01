import time
from datetime import datetime
from pathlib import Path
from subprocess import Popen
from typing import Optional

from adapters.ds.sinks.always_on_rtsp.config import Config
from adapters.ds.sinks.always_on_rtsp.input_pipeline import build_input_pipeline
from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrame
from adapters.ds.sinks.always_on_rtsp.output_pipeline import build_output_pipeline
from adapters.ds.sinks.always_on_rtsp.pipeline import PipelineThread
from savant.deepstream.encoding import check_encoder_is_available
from savant.gstreamer import Gst
from savant.gstreamer.codecs import Codec
from savant.gstreamer.element_factory import GstElementFactory
from savant.utils.logging import get_logger, init_logging

LOGGER_NAME = 'adapters.ao_sink'
logger = get_logger(LOGGER_NAME)


def run_ao_sink():
    init_logging()
    config = Config()

    if config.dev_mode:
        mediamtx_process = Popen(
            [
                '/opt/savant/mediamtx/mediamtx',
                str((Path(__file__).parent / 'mediamtx.yml').absolute()),
            ]
        )
        logger.info('Started MediaMTX, PID: %s', mediamtx_process.pid)
        assert (
            mediamtx_process.returncode is None
        ), f'Failed to start MediaMTX. Exit code: {mediamtx_process.returncode}.'
    else:
        mediamtx_process = None

    last_frame = LastFrame(timestamp=datetime.utcfromtimestamp(0))

    Gst.init(None)
    if not check_encoder_is_available(
        {'output_frame': {'codec': Codec.H264.value.name}}
    ):
        return

    logger.info('Starting Always-On-RTSP sink')
    factory = GstElementFactory()
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
        ao_sink_loop(output_pipeline_thread, input_pipeline_thread, mediamtx_process)
    except KeyboardInterrupt:
        pass
    logger.info('Stopping Always-On-RTSP sink')
    input_pipeline_thread.stop()
    output_pipeline_thread.stop()
    if mediamtx_process is not None:
        if mediamtx_process.returncode is None:
            logger.info('Terminating MediaMTX')
            mediamtx_process.terminate()
        logger.info('MediaMTX terminated. Exit code: %s.', mediamtx_process.returncode)
    logger.info('Always-On-RTSP sink stopped')


def ao_sink_loop(
    output_pipeline_thread: PipelineThread,
    input_pipeline_thread: PipelineThread,
    mediamtx_process: Optional[Popen],
):
    while output_pipeline_thread.is_running:
        input_pipeline_thread.start()
        while input_pipeline_thread.is_running and output_pipeline_thread.is_running:
            if mediamtx_process is not None and mediamtx_process.returncode is not None:
                logger.error(
                    'MediaMTX exited. Exit code: %s.', mediamtx_process.returncode
                )
                return
            time.sleep(1)
