#!/usr/bin/env python3
import signal
import time

from savant.gstreamer import Gst
from savant.gstreamer.runner import GstPipelineRunner
from savant.utils.config import opt_config, req_config, strtobool
from savant.utils.logging import get_logger, init_logging
from savant.utils.platform import is_aarch64
from savant.utils.welcome import get_starting_message

LOGGER_NAME = 'adapters.display_sink'

DEFAULT_SOURCE_TIMEOUT = 10
DEFAULT_SOURCE_EVICTION_INTERVAL = 1


class Config:
    def __init__(self):
        self.zmq_endpoint = req_config('ZMQ_ENDPOINT')
        self.zmq_type = opt_config('ZMQ_TYPE', 'SUB')
        self.zmq_bind = opt_config('ZMQ_BIND', False, strtobool)
        self.source_id = opt_config('SOURCE_ID')
        self.source_id_prefix = opt_config('SOURCE_ID_PREFIX')
        self.sync_input = opt_config('SYNC_INPUT', False, strtobool)
        self.closing_delay = opt_config('CLOSING_DELAY', 0, int)
        self.source_timeout = opt_config('SOURCE_TIMEOUT', DEFAULT_SOURCE_TIMEOUT, int)
        self.source_eviction_interval = opt_config(
            'SOURCE_EVICTION_INTERVAL',
            DEFAULT_SOURCE_EVICTION_INTERVAL,
            int,
        )
        self.ingress_queue_length = opt_config('INGRESS_QUEUE_LENGTH', None, int)
        self.ingress_queue_byte_size = opt_config('INGRESS_QUEUE_BYTE_SIZE', None, int)
        self.decoder_queue_length = opt_config('DECODER_QUEUE_LENGTH', None, int)
        self.decoder_queue_byte_size = opt_config('DECODER_QUEUE_BYTE_SIZE', None, int)
        self.egress_queue_length = opt_config('EGRESS_QUEUE_LENGTH', None, int)
        self.egress_queue_byte_size = opt_config('EGRESS_QUEUE_BYTE_SIZE', None, int)


def main():
    init_logging()
    # To gracefully shut down the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    config = Config()

    logger = get_logger(LOGGER_NAME)
    logger.info(get_starting_message('display sink adapter'))

    Gst.init(None)

    # nvjpegdec decoder is selected in decodebin according to the rank, but
    # the plugin doesn't support some jpg
    #  https://forums.developer.nvidia.com/t/nvvideoconvert-memory-compatibility-error/226138;
    # Set the rank to NONE for the plugin to not use it.
    # Decodebin will use nvv4l2decoder instead.
    if is_aarch64():
        factory = Gst.ElementFactory.find('nvjpegdec')
        if factory is not None:
            factory.set_rank(Gst.Rank.NONE)

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
    savant_rs_video_player.set_property('source-timeout', config.source_timeout)
    savant_rs_video_player.set_property(
        'source-eviction-interval', config.source_eviction_interval
    )
    if config.ingress_queue_length is not None:
        savant_rs_video_player.set_property(
            'ingress-queue-length', config.ingress_queue_length
        )
    if config.ingress_queue_byte_size is not None:
        savant_rs_video_player.set_property(
            'ingress-queue-byte-size', config.ingress_queue_byte_size
        )
    if config.decoder_queue_length is not None:
        savant_rs_video_player.set_property(
            'decoder-queue-length', config.decoder_queue_length
        )
    if config.decoder_queue_byte_size is not None:
        savant_rs_video_player.set_property(
            'decoder-queue-byte-size', config.decoder_queue_byte_size
        )
    if config.egress_queue_length is not None:
        savant_rs_video_player.set_property(
            'egress-queue-length', config.egress_queue_length
        )
    if config.egress_queue_byte_size is not None:
        savant_rs_video_player.set_property(
            'egress-queue-byte-size', config.egress_queue_byte_size
        )

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
