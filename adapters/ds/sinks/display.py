#!/usr/bin/env python3
import signal
import time

from savant.gstreamer import Gst
from savant.gstreamer.runner import GstPipelineRunner
from savant.utils.config import opt_config, req_config, strtobool
from savant.utils.logging import get_logger, init_logging
from savant.utils.welcome import get_starting_message

LOGGER_NAME = 'adapters.display_sink'


class Config:
    def __init__(self):
        self.zmq_endpoint = req_config('ZMQ_ENDPOINT')
        self.zmq_type = opt_config('ZMQ_TYPE', 'SUB')
        self.zmq_bind = opt_config('ZMQ_BIND', False, strtobool)
        self.source_id = opt_config('SOURCE_ID')
        self.source_id_prefix = opt_config('SOURCE_ID_PREFIX')
        self.sync_input = opt_config('SYNC_INPUT', False, strtobool)
        self.closing_delay = opt_config('CLOSING_DELAY', 0, int)


def main():
    init_logging()
    # To gracefully shut down the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    config = Config()

    logger = get_logger(LOGGER_NAME)
    logger.info(get_starting_message('display sink adapter'))

    Gst.init(None)

    pipeline: Gst.Pipeline = Gst.Pipeline.new()
    savant_rs_video_player: Gst.Element = Gst.ElementFactory.make(
        'savant_rs_video_player'
    )
    savant_rs_video_player.set_property('socket', config.zmq_endpoint)
    savant_rs_video_player.set_property('socket-type', config.zmq_type)
    savant_rs_video_player.set_property('bind', config.zmq_bind)
    if config.source_id:
        savant_rs_video_player.set_property('source-id', config.source_id)
    if config.source_id_prefix:
        savant_rs_video_player.set_property('source-id-prefix', config.source_id_prefix)
    savant_rs_video_player.set_property('sync', config.sync_input)
    savant_rs_video_player.set_property('closing-delay', config.closing_delay)
    pipeline.add(savant_rs_video_player)

    logger.info('Display sink started')
    with GstPipelineRunner(pipeline) as runner:
        try:
            while runner.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info('Interrupted')
            runner.shutdown()
    logger.info('Display sink stopped')


if __name__ == '__main__':
    main()
