import os
from distutils.util import strtobool
from pathlib import Path
from subprocess import Popen, TimeoutExpired
from typing import Dict, List, Optional

from adapters.ds.sinks.always_on_rtsp.config import opt_config
from savant.deepstream.encoding import check_encoder_is_available
from savant.gstreamer import Gst
from savant.gstreamer.codecs import Codec
from savant.utils.logging import get_logger, init_logging

LOGGER_NAME = 'adapters.ao_sink.entrypoint'
logger = get_logger(LOGGER_NAME)


class Config:
    def __init__(self):
        self.dev_mode = opt_config('DEV_MODE', False, strtobool)
        self.source_id: Optional[str] = opt_config('SOURCE_ID')
        self.source_ids: Optional[List[str]] = opt_config('SOURCE_IDS', '').split(',')
        assert (
            self.source_id or self.source_ids
        ), 'Either "SOURCE_ID" or "SOURCE_IDS" must be set.'

        self.rtsp_uri = opt_config('RTSP_URI')
        if self.dev_mode:
            assert (
                self.rtsp_uri is None
            ), '"RTSP_URI" cannot be set when "DEV_MODE=True"'
            self.rtsp_uri = 'rtsp://localhost:554/stream'
        else:
            assert (
                self.rtsp_uri is not None
            ), '"RTSP_URI" must be set when "DEV_MODE=False"'


def main():
    init_logging()
    config = Config()

    Gst.init(None)
    if not check_encoder_is_available(
        {'output_frame': {'codec': Codec.H264.value.name}}
    ):
        return

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

    if config.source_id:
        ao_sink_processes = {
            config.source_id: run_ao_sink_process(config.source_id, config.rtsp_uri)
        }
    else:
        ao_sink_processes = {
            source_id: run_ao_sink_process(
                source_id,
                f'{config.rtsp_uri.rstrip("/")}/{source_id}',
            )
            for source_id in config.source_ids
        }

    try:
        main_loop(ao_sink_processes, mediamtx_process)
    except KeyboardInterrupt:
        pass

    for source_id, ao_sink_process in ao_sink_processes.items():
        if ao_sink_process.returncode is None:
            logger.info('Terminating Always-On-RTSP for source %s', source_id)
            ao_sink_process.terminate()
        logger.info(
            'Always-On-RTSP for source %s terminated. Exit code: %s.',
            source_id,
            ao_sink_process.returncode,
        )

    if mediamtx_process is not None:
        if mediamtx_process.returncode is None:
            logger.info('Terminating MediaMTX')
            mediamtx_process.terminate()
        logger.info('MediaMTX terminated. Exit code: %s.', mediamtx_process.returncode)


def run_ao_sink_process(source_id: str, rtsp_uri: str):
    ao_sink_process = Popen(
        ['python', '-m', 'adapters.ds.sinks.always_on_rtsp.ao_sink'],
        env={**os.environ, 'SOURCE_ID': source_id, 'RTSP_URI': rtsp_uri},
    )
    logger.info('Started Always-On-RTSP, PID: %s', ao_sink_process.pid)
    if ao_sink_process.returncode is not None:
        raise RuntimeError(
            f'Failed to start Always-On-RTSP. Exit code: {ao_sink_process.returncode}.'
        )
    return ao_sink_process


def main_loop(
    ao_sink_processes: Dict[str, Popen],
    mediamtx_process: Optional[Popen],
):
    while True:
        for source_id, ao_sink_process in ao_sink_processes.items():
            try:
                returncode = ao_sink_process.wait(1)
                logger.error(
                    'Always-On-RTSP for source %s exited. Exit code: %s.',
                    source_id,
                    returncode,
                )
                return
            except TimeoutExpired:
                pass

        if mediamtx_process is not None:
            try:
                returncode = mediamtx_process.wait(1)
                logger.error('MediaMTX exited. Exit code: %s.', returncode)
                return
            except TimeoutExpired:
                pass


if __name__ == '__main__':
    main()
