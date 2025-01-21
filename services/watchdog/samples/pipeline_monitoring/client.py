#!/usr/bin/env python3
import os
import random
import signal
import sys
import time

from savant.client import SinkBuilder
from savant.utils import logging


def main():
    # To gracefully shut down the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    logging.init_logging()
    logger = logging.get_logger('Client')

    zmq_sink_endpoint = os.environ.get('ZMQ_SINK_ENDPOINT')
    min_sleep = int(os.environ.get('MIN_SLEEP', 0))
    max_sleep = int(os.environ.get('MAX_SLEEP', 200))

    if not zmq_sink_endpoint:
        sys.exit(
            'ZMQ_SINK_ENDPOINT is not set. Please provide the ZMQ_SINK_ENDPOINT in the environment variable.'
        )

    # Build the sink
    sink = SinkBuilder().with_socket(zmq_sink_endpoint).build()

    for result in sink:
        sleep_duration = random.uniform(min_sleep, max_sleep)
        if sleep_duration > 0:
            logger.info('Stop processing for %s seconds', sleep_duration)
            time.sleep(sleep_duration)

        if result is not None:
            logger.debug(
                'Received frame %s/%s (keyframe=%s)',
                result.frame_meta.source_id,
                result.frame_meta.pts,
                result.frame_meta.keyframe,
            )
        else:
            source_id = result.eos.source_id
            logger.debug('Received EOS for source %s', source_id)


if __name__ == '__main__':
    main()
