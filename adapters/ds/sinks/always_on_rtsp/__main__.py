"""Entrypoint for Always-On-RTSP sink."""
import signal
import time
from pathlib import Path
from subprocess import Popen
from threading import Thread
from typing import Optional

from adapters.ds.sinks.always_on_rtsp.api import Api
from adapters.ds.sinks.always_on_rtsp.app_config import AppConfig
from adapters.ds.sinks.always_on_rtsp.stream_manager import Stream, StreamManager
from adapters.ds.sinks.always_on_rtsp.utils import (
    nvidia_runtime_is_available,
    process_is_alive,
)
from adapters.ds.sinks.always_on_rtsp.zeromq_proxy import ZeroMqProxy
from savant.gstreamer import Gst
from savant.utils.logging import get_logger, init_logging

LOGGER_NAME = 'adapters.ao_sink.entrypoint'
logger = get_logger(LOGGER_NAME)


def main():
    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    init_logging()
    config = AppConfig()

    Gst.init(None)
    if nvidia_runtime_is_available():
        logger.info(
            'NVIDIA runtime is available. Using hardware-based decoding/encoding.'
        )
        if not config.codec.value.nv_encoder:
            logger.error(
                'Hardware-based encoding is not available for codec %s.',
                config.codec.value.name,
            )
            return

        from savant.utils.check_display import check_display_env

        check_display_env(logger)

        from savant.deepstream.encoding import check_encoder_is_available

        if not check_encoder_is_available(
            {'output_frame': {'codec': config.codec.value.name}}
        ):
            return
    else:
        logger.warning(
            'You are using the AO-RTSP adapter with a software-based decoder/encoder '
            '(without NVDEC/NVENC). This mode must be used only when hardware-based '
            'encoding is not available (Jetson Orin Nano, A100, H100). '
            'If the hardware-based encoding is available, run the adapter with Nvidia '
            'runtime enabled to activate hardware-based decoding/encoding.'
        )
        if not config.codec.value.sw_encoder:
            logger.error(
                'Software-based encoding is not available for codec %s.',
                config.codec.value.name,
            )
            return

    internal_socket = 'ipc:///tmp/ao-sink-internal-socket.ipc'
    zmq_reader_endpoint = f'sub+connect:{internal_socket}'
    zmq_proxy = ZeroMqProxy(
        input_socket=config.zmq_endpoint,
        input_socket_type=config.zmq_socket_type,
        input_bind=config.zmq_socket_bind,
        output_socket=f'pub+bind:{internal_socket}',
    )
    zmq_proxy.start()
    zmq_proxy_thread = Thread(target=zmq_proxy.run, daemon=True)
    zmq_proxy_thread.start()

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

    stream_manager = StreamManager(config, zmq_reader_endpoint)
    stream_manager.start()

    for source_id in config.source_ids:
        stream_manager.add_stream(source_id, Stream())

    api = Api(config, stream_manager)
    api.start()

    try:
        main_loop(stream_manager, api, mediamtx_process)
    except KeyboardInterrupt:
        pass

    stream_manager.stop()
    for source_id in stream_manager.get_all_streams().keys():
        stream_manager.delete_stream(source_id)

    if mediamtx_process is not None:
        if mediamtx_process.returncode is None:
            logger.info('Terminating MediaMTX')
            mediamtx_process.terminate()
            mediamtx_process.wait()
        logger.info('MediaMTX terminated. Exit code: %s.', mediamtx_process.returncode)


def main_loop(
    stream_manager: StreamManager,
    api: Api,
    mediamtx_process: Optional[Popen],
):
    while True:
        for source_id, stream in stream_manager.get_all_streams().items():
            if stream.exit_code is not None:
                return

        if mediamtx_process is not None:
            exit_code = process_is_alive(mediamtx_process)
            if exit_code is not None:
                logger.error('MediaMTX exited. Exit code: %s.', exit_code)
                return

        if not api.is_alive():
            logger.error('API server is not running.')
            return

        time.sleep(1)  # TODO: configure


if __name__ == '__main__':
    main()
