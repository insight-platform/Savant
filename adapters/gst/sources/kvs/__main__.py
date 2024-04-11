import signal
import time

from savant.gstreamer import Gst
from savant.utils.logging import get_logger, init_logging

from . import LOGGER_PREFIX
from .api import Api
from .config import Config
from .stream_manager import StreamManager


def main():
    init_logging()
    logger = get_logger(LOGGER_PREFIX)
    logger.info('Starting the adapter')

    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    config = Config()

    Gst.init(None)

    stream_manager = StreamManager(config)
    api = Api(config, stream_manager)

    stream_manager.start()
    api.start()
    try:
        while api.is_running() and stream_manager.is_running():
            time.sleep(1)
    finally:
        logger.info('Stopping the adapter')
        stream_manager.stop()


if __name__ == '__main__':
    main()
