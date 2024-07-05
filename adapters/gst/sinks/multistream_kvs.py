#!/usr/bin/env python3
import itertools
import os
import signal
import time
from fractions import Fraction
from typing import Dict, Optional
from savant.utils.config import req_config

from pygstsavantframemeta import (
    gst_buffer_add_savant_frame_meta,
    gst_buffer_get_savant_frame_meta,
)
from pykvssdk import KvsWrapper, configure_logging
from savant_rs.pipeline2 import (
    StageFunction,
    VideoPipeline,
    VideoPipelineConfiguration,
    VideoPipelineStagePayloadType,
)
from savant_rs.primitives import EndOfStream, VideoFrame

from adapters.python.sinks.chunk_writer import ChunkWriter
from gst_plugins.python.savant_rs_video_demux_common import FrameParams, build_caps
from savant.api.enums import ExternalFrameType
from savant.gstreamer import Gst, GstApp
from savant.gstreamer.codecs import Codec
from savant.utils.config import opt_config, strtobool
from savant.utils.logging import get_logger, init_logging
from savant.utils.zeromq import ZeroMQMessage, ZeroMQSource

LOGGER_PREFIX = 'adapters.multistream_kvs_sink'

KVS_LOG_CONFIG = os.path.normpath(
    os.path.join(os.path.dirname(__file__), 'kvs_log_configuration')
)
KVS_LOG_CONFIG_TEMPLATE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), 'kvs_log_configuration.template')
)

CODEC_TO_CAPS = {
    Codec.H264: 'video/x-h264,stream-format=avc,alignment=au',
    Codec.HEVC: 'video/x-h265,stream-format=hvc1,alignment=au',
}


class AwsConfig:
    def __init__(self):
        self.region = req_config('AWS_REGION')
        self.access_key = req_config('AWS_ACCESS_KEY')
        self.secret_key = req_config('AWS_SECRET_KEY')


class ZmqConfig:
    def __init__(self):
        self.endpoint = req_config('ZMQ_ENDPOINT')
        self.source_id = opt_config('SOURCE_ID')
        self.source_id_prefix = opt_config('SOURCE_ID_PREFIX')


class BufferConfig:
    def __init__(self):
        self.low_threshold = opt_config('BUFFER_LOW_THRESHOLD', 30, int)
        self.high_threshold = opt_config('BUFFER_HIGH_THRESHOLD', 40, int)
        assert (
            self.low_threshold < self.high_threshold
        ), 'BUFFER_LOW_THRESHOLD must be less than BUFFER_HIGH_THRESHOLD'


class FpsMeterConfig:
    def __init__(self):
        self.period_seconds = opt_config('FPS_PERIOD_SECONDS', None, float)
        self.period_frames = opt_config('FPS_PERIOD_FRAMES', 1000, int)


class Config:
    def __init__(self):
        self.allow_create_stream = opt_config('ALLOW_CREATE_STREAM', False, strtobool)
        self.stream_name_prefix = opt_config('STREAM_NAME_PREFIX', '')
        self.kvssdk_loglevel = os.environ.get('KVSSDK_LOGLEVEL', 'INFO')
        self.zmq: ZmqConfig = ZmqConfig()
        self.aws: AwsConfig = AwsConfig()
        self.buffer: BufferConfig = BufferConfig()
        self.fps_meter: FpsMeterConfig = FpsMeterConfig()


class KvsWriter(ChunkWriter):
    def __init__(
        self,
        source_id: str,
        kvs_name: str,
        frame_params: FrameParams,
        video_pipeline: VideoPipeline,
        config: Config,
    ):
        self.source_id = source_id
        self.kvs_name = kvs_name
        self.frame_params = frame_params
        self.video_pipeline = video_pipeline
        self.config = config

        self.kvs: KvsWrapper = KvsWrapper(
            config.aws.region,
            config.aws.access_key,
            config.aws.secret_key,
            self.kvs_name,
            frame_params.codec.value.name,
            config.allow_create_stream,
            round(Fraction(frame_params.framerate)),
            config.buffer.low_threshold,
            config.buffer.high_threshold,
        )
        self.stream_started = False

        self.pipeline: Optional[Gst.Pipeline] = None
        self.appsrc: Optional[GstApp.AppSrc] = None
        self.appsink: Optional[GstApp.AppSink] = None

        self.frame_idx_gen = itertools.count()
        super().__init__(chunk_size=0, logger_prefix=LOGGER_PREFIX)

    def _on_buffer(self, sink: GstApp.AppSink) -> Gst.FlowReturn:
        """Receives a buffer from Gst pipeline and sends it to KVS."""

        self.logger.debug('Received buffer from source %s', self.source_id)
        sample: Gst.Sample = sink.emit('pull-sample')
        if not self.stream_started:
            self.logger.info(
                'Starting stream %s for source %s',
                self.kvs_name,
                self.source_id,
            )
            caps: Gst.Caps = sample.get_caps()
            codec_data_buffer: Gst.Buffer = caps.get_structure(0).get_value(
                'codec_data'
            )
            codec_data: bytes = codec_data_buffer.extract_dup(
                0, codec_data_buffer.get_size()
            )
            if not self.kvs.start(codec_data, len(codec_data)):
                self.logger.error(
                    'Failed to start stream %s',
                    self.kvs_name,
                )
                return Gst.FlowReturn.ERROR
            self.logger.info(
                'Stream %s for source %s started',
                self.kvs_name,
                self.source_id,
            )
            self.stream_started = True

        buffer: Gst.Buffer = sample.get_buffer()
        savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
        if savant_frame_meta is None:
            self.logger.warning(
                'No Savant Frame Metadata found on buffer with PTS %s, skipping.',
                buffer.pts,
            )
            return Gst.FlowReturn.OK
        video_frame, _ = self.video_pipeline.get_independent_frame(
            savant_frame_meta.idx
        )
        self.video_pipeline.delete(savant_frame_meta.idx)

        frame_data: bytes = buffer.extract_dup(0, buffer.get_size())
        self.logger.debug(
            'Sending frame with pts=%s to %s',
            buffer.pts,
            self.kvs_name,
        )
        self.kvs.put_frame(
            frame_data,
            len(frame_data),
            savant_frame_meta.idx,
            buffer.pts,
            buffer.dts,
            buffer.duration,
            not buffer.has_flags(Gst.BufferFlags.DELTA_UNIT),
            video_frame.uuid,
        )

        return Gst.FlowReturn.OK

    def _on_eos(self, sink: GstApp.AppSink):
        """Stop the stream when EOS is received."""

        self.logger.info(
            'Received EOS from source %s',
            self.source_id,
        )
        self.kvs.stop_sync()
        self.logger.info(
            'Stream %s for source %s stopped',
            self.kvs_name,
            self.source_id,
        )
        self.pipeline = None

    def _write_video_frame(
        self,
        frame: VideoFrame,
        data: Optional[bytes],
        frame_num: int,
    ) -> bool:
        if not data:
            return True

        self.logger.debug(
            'Processing frame with pts=%s to %s',
            frame.pts,
            self.kvs_name,
        )

        buffer: Gst.Buffer = Gst.Buffer.new_wrapped(data)
        buffer.pts = frame.pts
        buffer.dts = frame.dts
        buffer.duration = frame.duration
        if not frame.keyframe:
            buffer.set_flags(Gst.BufferFlags.DELTA_UNIT)
        frame_idx = self.video_pipeline.add_frame('kvs-sink', frame)
        gst_buffer_add_savant_frame_meta(buffer, frame_idx)

        ret = self.appsrc.push_buffer(buffer)
        self.logger.debug(
            'Processing frame with pts=%s to %s: %s',
            frame.pts,
            self.kvs_name,
            ret,
        )

        return ret == Gst.FlowReturn.OK

    def _write_eos(self, eos: EndOfStream) -> bool:
        self.close()
        return True

    def _open(self):
        while self.pipeline is not None:
            self.logger.debug(
                'Waiting for the previous pipeline to be closed for chunk %s of source %s',
                self.chunk_idx,
                self.source_id,
            )
            time.sleep(0.1)

        elements = [
            'appsrc name=appsrc emit-signals=false format=time max-buffers=1 block=true',
            f'{self.frame_params.codec.value.parser} config-interval=-1',
            CODEC_TO_CAPS[self.frame_params.codec],
            'appsink name=appsink emit-signals=true sync=false max-buffers=1',
        ]

        self.pipeline = Gst.parse_launch(' ! '.join(elements))
        self.appsrc = self.pipeline.get_by_name('appsrc')
        self.appsink = self.pipeline.get_by_name('appsink')
        self.appsrc.set_caps(build_caps(self.frame_params))
        self.appsink.connect('new-sample', self._on_buffer)
        self.appsink.connect('eos', self._on_eos)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _close(self):
        self.logger.debug(
            'Stopping and removing sink elements for chunk %s of source %s',
            self.chunk_idx,
            self.source_id,
        )
        self.appsrc.end_of_stream()


class MultiStreamKvsSink:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(f'{LOGGER_PREFIX}.{self.__class__.__name__}')
        self.writers: Dict[str, ChunkWriter] = {}
        self.video_pipeline: VideoPipeline = self.build_video_pipeline()

    def build_video_pipeline(self) -> VideoPipeline:
        conf = VideoPipelineConfiguration()
        conf.frame_period = self.config.fps_meter.period_frames
        conf.timestamp_period = (
            int(self.config.fps_meter.period_seconds * 1000)
            if self.config.fps_meter.period_seconds
            else None
        )

        return VideoPipeline(
            'kvs-sink',
            [
                (
                    'kvs-sink',
                    VideoPipelineStagePayloadType.Frame,
                    StageFunction.none(),
                    StageFunction.none(),
                ),
            ],
            conf,
        )

    def write(self, zmq_message: ZeroMQMessage):
        message = zmq_message.message
        message.validate_seq_id()
        if message.is_video_frame():
            return self._write_video_frame(
                message.as_video_frame(),
                zmq_message.content,
            )
        if message.is_end_of_stream():
            return self._write_eos(message.as_end_of_stream())
        self.logger.debug('Unsupported message type for message %r', message)

    def _write_video_frame(self, frame: VideoFrame, content: Optional[bytes]) -> bool:
        frame_params = FrameParams.from_video_frame(frame)
        if frame_params.codec not in [Codec.H264, Codec.HEVC]:
            self.logger.warning(
                'Received frame %s with unsupported codec %s. Skipping it.',
                frame.pts,
                frame_params.codec,
            )
            return True

        if frame.content.is_none():
            self.logger.debug(
                'Received frame %s from source %s is empty. Skipping it.',
                frame.pts,
                frame.source_id,
            )
            return True

        if frame.content.is_internal():
            content = frame.content.get_data_as_bytes()
            self.logger.debug(
                'Received frame %s from source %s, size: %s bytes',
                frame.pts,
                frame.source_id,
                len(content),
            )
        else:
            frame_type = ExternalFrameType(frame.content.get_method())
            if frame_type != ExternalFrameType.ZEROMQ:
                self.logger.error('Unsupported frame type "%s".', frame_type.value)
                return True
            if not content:
                self.logger.error(
                    'Frame content is missing. Skipping frame %s from source %s.',
                    frame.pts,
                    frame.source_id,
                )
                return True

        writer = self.writers.get(frame.source_id)
        if writer is None:
            writer = KvsWriter(
                source_id=frame.source_id,
                kvs_name=self.config.stream_name_prefix + frame.source_id,
                frame_params=frame_params,
                video_pipeline=self.video_pipeline,
                config=self.config,
            )
            self.writers[frame.source_id] = writer

        return writer.write_video_frame(frame, content, frame.keyframe)

    def _write_eos(self, eos: EndOfStream):
        self.logger.info('Received EOS from source %s.', eos.source_id)
        writer = self.writers.get(eos.source_id)
        if writer is None:
            return False
        writer.write_eos(eos)
        writer.flush()
        return True

    def terminate(self):
        for file_writer in self.writers.values():
            file_writer.close()


def main():
    init_logging()
    logger = get_logger(LOGGER_PREFIX)

    # To gracefully shutdown the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    config = Config()

    with open(KVS_LOG_CONFIG_TEMPLATE) as f:
        log_config = f.read()
    log_config = log_config.replace('[[LOGLEVEL]]', config.kvssdk_loglevel)
    with open(KVS_LOG_CONFIG, 'w') as f:
        f.write(log_config)
    configure_logging(KVS_LOG_CONFIG)

    Gst.init(None)

    # possible exceptions will cause app to crash and log error by default
    # no need to handle exceptions here
    source = ZeroMQSource(
        config.zmq.endpoint,
        source_id=config.zmq.source_id,
        source_id_prefix=config.zmq.source_id_prefix,
    )

    sink = MultiStreamKvsSink(config)
    logger.info('Multistream KVS sink started')

    try:
        source.start()
        for zmq_message in source:
            sink.write(zmq_message)
    except KeyboardInterrupt:
        logger.info('Interrupted')
    finally:
        source.terminate()
        sink.terminate()


if __name__ == '__main__':
    main()
