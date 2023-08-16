import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from distutils.util import strtobool
from pathlib import Path
from subprocess import Popen
from threading import Thread
from typing import Callable, List, Optional

import pyds
from pygstsavantframemeta import gst_buffer_get_savant_frame_meta

from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrame
from savant.config.schema import PipelineElement
from savant.deepstream.encoding import check_encoder_is_available
from savant.gstreamer import Gst
from savant.gstreamer.codecs import CODEC_BY_CAPS_NAME, Codec
from savant.gstreamer.element_factory import GstElementFactory
from savant.gstreamer.metadata import metadata_pop_frame_meta
from savant.gstreamer.runner import GstPipelineRunner
from savant.utils.platform import is_aarch64
from savant.utils.zeromq import ReceiverSocketTypes
from savant.utils.logging import get_logger

LOGGER_NAME = 'ao_sink'
logger = get_logger(LOGGER_NAME)


def opt_config(name, default=None, convert=None):
    conf_str = os.environ.get(name)
    if conf_str:
        return convert(conf_str) if convert else conf_str
    return default


class Config:
    def __init__(self):
        self.dev_mode = opt_config('DEV_MODE', False, strtobool)

        self.stub_file_location = Path(os.environ['STUB_FILE_LOCATION'])
        if not self.stub_file_location.exists():
            raise RuntimeError(f'File {self.stub_file_location} does not exist.')
        if not self.stub_file_location.is_file():
            raise RuntimeError(f'{self.stub_file_location} is not a file.')

        self.max_delay_ms = opt_config('MAX_DELAY_MS', 1000, int)
        self.transfer_mode = opt_config('TRANSFER_MODE', 'scale-to-fit')
        self.source_id = os.environ['SOURCE_ID']

        self.zmq_endpoint = os.environ['ZMQ_ENDPOINT']
        self.zmq_socket_type = opt_config(
            'ZMQ_TYPE',
            ReceiverSocketTypes.SUB,
            ReceiverSocketTypes.__getitem__,
        )
        self.zmq_socket_bind = opt_config('ZMQ_BIND', False, strtobool)

        self.rtsp_uri = opt_config('RTSP_URI')
        if self.dev_mode:
            assert (
                self.rtsp_uri is None
            ), '"RTSP_URI" cannot be set when "DEV_MODE=True"'
            self.rtsp_uri = 'rtsp://localhost:554/stream'
        self.rtsp_protocols = opt_config('RTSP_PROTOCOLS', 'tcp')
        self.rtsp_latency_ms = opt_config('RTSP_LATENCY_MS', 100, int)
        self.rtsp_keep_alive = opt_config('RTSP_KEEP_ALIVE', True, strtobool)

        self.encoder_profile = opt_config('ENCODER_PROFILE', 'High')
        # default nvv4l2h264enc bitrate
        self.encoder_bitrate = opt_config('ENCODER_BITRATE', 4000000, int)

        self.fps_period_frames = opt_config('FPS_PERIOD_FRAMES', 1000, int)
        self.fps_period_seconds = opt_config('FPS_PERIOD_SECONDS', convert=float)
        self.fps_output = opt_config('FPS_OUTPUT', 'stdout')

        self.metadata_output = opt_config('METADATA_OUTPUT')

        self.framerate = opt_config('FRAMERATE', '30/1')
        self.sync = opt_config('SYNC_OUTPUT', False, strtobool)

    @property
    def fps_meter_properties(self):
        props = {'output': self.fps_output}
        if self.fps_period_seconds:
            props['period-seconds'] = self.fps_period_seconds
        else:
            props['period-frames'] = self.fps_period_frames
        return props

    @property
    def nvvideoconvert_properties(self):
        props = {}
        if not is_aarch64():
            props['nvbuf-memory-type'] = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        return props


def log_frame_metadata(pad: Gst.Pad, info: Gst.PadProbeInfo, config: Config):
    buffer: Gst.Buffer = info.get_buffer()
    savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
    frame_idx = savant_frame_meta.idx if savant_frame_meta else None
    frame_pts = buffer.pts
    metadata = metadata_pop_frame_meta(config.source_id, frame_idx, frame_pts)
    metadata_json = json.dumps(asdict(metadata), default=dict)
    if config.metadata_output == 'logger':
        logger.info('Frame metadata: %s', metadata_json)
    else:
        print(f'Frame metadata: {metadata_json}')
    return Gst.PadProbeReturn.OK


def link_added_pad(
    element: Gst.Element,
    src_pad: Gst.Pad,
    sink_pad: Gst.Pad,
):
    assert src_pad.link(sink_pad) == Gst.PadLinkReturn.OK


def on_demuxer_pad_added(
    element: Gst.Element,
    src_pad: Gst.Pad,
    config: Config,
    pipeline: Gst.Pipeline,
    factory: GstElementFactory,
    sink_pad: Gst.Pad,
):
    caps: Gst.Caps = src_pad.get_pad_template_caps()
    logger.debug(
        'Added pad %s on element %s. Caps: %s.',
        src_pad.get_name(),
        element.get_name(),
        caps,
    )
    codec = CODEC_BY_CAPS_NAME[caps[0].get_name()]
    if config.metadata_output:
        src_pad.add_probe(Gst.PadProbeType.BUFFER, log_frame_metadata, config)

    if codec == Codec.RAW_RGBA:
        capsfilter = factory.create(
            PipelineElement(
                'capsfilter',
                properties={'caps': caps},
            )
        )
        pipeline.add(capsfilter)
        assert capsfilter.get_static_pad('src').link(sink_pad) == Gst.PadLinkReturn.OK
        assert src_pad.link(capsfilter.get_static_pad('sink')) == Gst.PadLinkReturn.OK
        capsfilter.sync_state_with_parent()
    else:
        decodebin = factory.create(PipelineElement('decodebin'))
        pipeline.add(decodebin)
        decodebin_sink_pad: Gst.Pad = decodebin.get_static_pad('sink')
        decodebin.connect('pad-added', link_added_pad, sink_pad)
        assert src_pad.link(decodebin_sink_pad) == Gst.PadLinkReturn.OK
        decodebin.sync_state_with_parent()
        logger.debug('Added decoder %s.', decodebin.get_name())


def build_input_pipeline(
    config: Config,
    last_frame: LastFrame,
    factory: GstElementFactory,
):
    pipeline: Gst.Pipeline = Gst.Pipeline.new('input-pipeline')

    source_elements = [
        PipelineElement(
            'zeromq_src',
            properties={
                'socket': config.zmq_endpoint,
                'socket-type': config.zmq_socket_type.name,
                'bind': config.zmq_socket_bind,
            },
        ),
        PipelineElement(
            'savant_rs_video_demux',
            properties={
                'source-id': config.source_id,
                'store-metadata': bool(config.metadata_output),
            },
        ),
    ]
    sink_elements = [
        PipelineElement(
            'nvvideoconvert',
            properties=config.nvvideoconvert_properties,
        ),
        PipelineElement(
            'capsfilter',
            properties={'caps': 'video/x-raw(memory:NVMM), format=RGBA'},
        ),
        PipelineElement(
            'fps_meter',
            properties=config.fps_meter_properties,
        ),
    ]
    if config.sync:
        sink_elements.append(
            PipelineElement(
                'adjust_timestamps',
                properties={'adjust-first-frame': True},
            )
        )
    sink_elements.append(
        PipelineElement(
            'always_on_rtsp_frame_sink',
            properties={
                'last-frame': last_frame,
                'sync': config.sync,
            },
        )
    )

    gst_source_elements = add_elements(pipeline, source_elements, factory)
    gst_sink_elements = add_elements(pipeline, sink_elements, factory)
    savant_rs_video_demux = gst_source_elements[-1]
    nvvideoconvert = gst_sink_elements[0]

    savant_rs_video_demux.connect(
        'pad-added',
        on_demuxer_pad_added,
        config,
        pipeline,
        factory,
        nvvideoconvert.get_static_pad('sink'),
    )

    return pipeline


def build_output_pipeline(
    config: Config,
    last_frame: LastFrame,
    factory: GstElementFactory,
) -> Gst.Pipeline:
    pipeline: Gst.Pipeline = Gst.Pipeline.new('output-pipeline')

    encoder_properties = {
        'profile': config.encoder_profile,
        'bitrate': config.encoder_bitrate,
    }
    if not is_aarch64():
        # nvv4l2h264enc doesn't encode video properly for the RTSP stream on dGPU
        # https://forums.developer.nvidia.com/t/rtsp-stream-sent-by-rtspclientsink-doesnt-play-in-deepstream-6-2/244194
        encoder_properties['tuning-info-id'] = 'HighQualityPreset'
    elements = [
        PipelineElement(
            'filesrc',
            properties={
                'location': str(config.stub_file_location.absolute()),
            },
        ),
        PipelineElement('jpegparse'),
        PipelineElement('jpegdec'),
        PipelineElement('imagefreeze'),
        PipelineElement(
            'nvvideoconvert',
            properties=config.nvvideoconvert_properties,
        ),
        PipelineElement(
            'capsfilter',
            properties={
                'caps': f'video/x-raw(memory:NVMM), format=RGBA, framerate={config.framerate}'
            },
        ),
        PipelineElement(
            'always_on_rtsp_frame_processor',
            properties={
                'max-delay-ms': config.max_delay_ms,
                'mode': config.transfer_mode,
                'last-frame': last_frame,
            },
        ),
        PipelineElement(
            'nvvideoconvert',
            properties=config.nvvideoconvert_properties,
        ),
        PipelineElement(
            'nvv4l2h264enc',
            properties=encoder_properties,
        ),
        PipelineElement(
            'h264parse',
            properties={
                'config-interval': -1,
            },
        ),
        PipelineElement(
            'fps_meter',
            properties=config.fps_meter_properties,
        ),
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

    add_elements(pipeline, elements, factory)

    return pipeline


def add_elements(
    pipeline: Gst.Pipeline,
    elements: List[PipelineElement],
    factory: GstElementFactory,
) -> List[Gst.Element]:
    gst_elements: List[Gst.Element] = []
    for element in elements:
        gst_element = factory.create(element)
        pipeline.add(gst_element)
        if gst_elements:
            assert gst_elements[-1].link(gst_element)
        gst_elements.append(gst_element)
    return gst_elements


class PipelineThread:
    def __init__(
        self,
        build_pipeline: Callable[[Config, LastFrame, GstElementFactory], Gst.Pipeline],
        thread_name: str,
        config: Config,
        last_frame: LastFrame,
        factory: GstElementFactory,
    ):
        self.build_pipeline = build_pipeline
        self.thread_name = thread_name
        self.config = config
        self.last_frame = last_frame
        self.factory = factory

        self.is_running = False
        self.thread: Optional[Thread] = None
        self.logger = get_logger(f'{LOGGER_NAME}.{self.__class__.__name__}')

    def start(self):
        self.is_running = True
        self.thread = Thread(name=self.thread_name, target=self.workload)
        self.thread.start()

    def stop(self):
        self.is_running = False

    def join(self):
        self.thread.join()

    def workload(self):
        pipeline = self.build_pipeline(self.config, self.last_frame, self.factory)
        self.logger.info('Starting pipeline %s', pipeline.get_name())
        with GstPipelineRunner(pipeline) as runner:
            while self.is_running and runner._is_running:
                time.sleep(1)
        self.logger.info('Pipeline %s is stopped', pipeline.get_name())
        self.is_running = False


def main():
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
        main_loop(output_pipeline_thread, input_pipeline_thread, mediamtx_process)
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


def main_loop(
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


if __name__ == '__main__':
    main()
