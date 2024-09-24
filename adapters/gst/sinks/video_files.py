#!/usr/bin/env python3
import os
import signal
import threading
from datetime import timedelta
from time import time
from typing import Dict, Optional

from savant_rs.primitives import EndOfStream, VideoFrame

from adapters.python.sinks.chunk_writer import ChunkWriter, CompositeChunkWriter
from adapters.python.sinks.metadata_json import (
    MetadataJsonSink,
    MetadataJsonWriter,
    Patterns,
    get_location,
    get_tag_location,
)
from gst_plugins.python.savant_rs_video_demux_common import FrameParams, build_caps
from savant.api.parser import convert_ts
from savant.gstreamer import GLib, Gst, GstApp
from savant.gstreamer.codecs import Codec
from savant.utils.config import opt_config, req_config, strtobool
from savant.utils.logging import get_logger, init_logging
from savant.utils.welcome import get_starting_message
from savant.utils.zeromq import ZeroMQMessage, ZeroMQSource

LOGGER_NAME = 'adapters.video_files_sink'
DEFAULT_CHUNK_SIZE = 10000


# Modified version of savant.gstreamer.runner.GstPipelineRunner
# to avoid unnecessary dependencies (omegaconf, opencv, etc.)
class GstPipelineRunner:
    """Manages running Gstreamer pipeline.

    :param pipeline: GstPipeline or Gst.Pipeline to run.
    :param shutdown_timeout: Seconds to wait for pipeline shutdown.
    """

    def __init__(
        self,
        pipeline: Gst.Pipeline,
        shutdown_timeout: int = 10,  # make configurable
    ):
        self._logger = get_logger(f'{LOGGER_NAME}.{self.__class__.__name__}')
        self._shutdown_timeout = shutdown_timeout

        # pipeline error storage
        self._error: Optional[str] = None

        # running pipeline flag
        self._is_running = False

        # pipeline execution start time, will be set on startup
        self._start_time = 0.0

        self._main_loop = GLib.MainLoop()
        self._main_loop_thread = threading.Thread(target=self._main_loop_run)

        self._pipeline: Gst.Pipeline = pipeline

    def _main_loop_run(self):
        try:
            self._main_loop.run()
        finally:
            self.shutdown()
            if self._error:
                raise RuntimeError(self._error)

    def startup(self):
        """Starts pipeline."""
        self._logger.info('Starting pipeline `%s`...', self._pipeline)
        start_time = time()

        bus = self._pipeline.get_bus()
        self._logger.debug('Adding signal watch and connecting callbacks...')
        bus.add_signal_watch()
        bus.connect('message::error', self.on_error)
        bus.connect('message::eos', self.on_eos)
        bus.connect('message::warning', self.on_warning)
        bus.connect('message::state-changed', self.on_state_changed)

        self._logger.debug('Setting pipeline to READY...')
        self._pipeline.set_state(Gst.State.READY)

        self._logger.debug('Setting pipeline to PLAYING...')
        self._pipeline.set_state(Gst.State.PLAYING)

        self._logger.debug('Starting main loop thread...')
        self._is_running = True
        self._main_loop_thread.start()

        end_time = time()
        exec_seconds = end_time - start_time
        self._logger.info(
            'The pipeline is initialized and ready to process data. Initialization took %s.',
            timedelta(seconds=exec_seconds),
        )

        self._start_time = end_time

    def shutdown(self):
        """Stops pipeline."""
        self._logger.debug('shutdown() called.')
        if not self._is_running:
            self._logger.debug('The pipeline is shutting down already.')
            return

        self._is_running = False

        if self._main_loop.is_running():
            self._logger.debug('Quitting main loop...')
            self._main_loop.quit()

        pipeline_state_thread = threading.Thread(
            target=self._pipeline.set_state,
            args=(Gst.State.NULL,),
            daemon=True,
        )
        self._logger.debug('Setting pipeline to NULL...')
        pipeline_state_thread.start()
        try:
            pipeline_state_thread.join(self._shutdown_timeout)
        except RuntimeError:
            self._logger.error('Failed to join thread.')

        exec_seconds = time() - self._start_time
        self._logger.info(
            'The pipeline is about to stop. Operation took %s.',
            timedelta(seconds=exec_seconds),
        )
        if pipeline_state_thread.is_alive():
            self._logger.warning('Pipeline shutdown timeout exceeded.')

    def on_error(  # pylint: disable=unused-argument
        self, bus: Gst.Bus, message: Gst.Message
    ):
        """Error callback."""
        err, debug = message.parse_error()
        # calling `raise` here causes the pipeline to hang,
        # just save message and handle it later
        self._error = self.build_error_message(message, err, debug)
        self._logger.error(self._error)
        self._error += f' Debug info: "{debug}".'
        self.shutdown()

    def build_error_message(self, message: Gst.Message, err: GLib.GError, debug: str):
        """Build error message."""
        return f'Received error "{err}" from {message.src.name}.'

    def on_eos(  # pylint: disable=unused-argument
        self, bus: Gst.Bus, message: Gst.Message
    ):
        """EOS callback."""
        self._logger.info('Received EOS from %s.', message.src.name)
        self.shutdown()

    def on_warning(  # pylint: disable=unused-argument
        self, bus: Gst.Bus, message: Gst.Message
    ):
        """Warning callback."""
        warn, debug = message.parse_warning()
        self._logger.warning('Received warning %s. %s', warn, debug)

    def on_state_changed(  # pylint: disable=unused-argument
        self, bus: Gst.Bus, msg: Gst.Message
    ):
        """Change state callback."""
        if not msg.src == self._pipeline:
            # not from the pipeline, ignore
            return

        old_state, new_state, _ = msg.parse_state_changed()
        old_state_name = Gst.Element.state_get_name(old_state)
        new_state_name = Gst.Element.state_get_name(new_state)
        self._logger.debug(
            'Pipeline state changed from %s to %s.', old_state_name, new_state_name
        )

        if old_state != new_state and os.getenv('GST_DEBUG_DUMP_DOT_DIR'):
            file_name = f'pipeline.{old_state_name}_{new_state_name}'
            Gst.debug_bin_to_dot_file_with_ts(
                self._pipeline, Gst.DebugGraphDetails.ALL, file_name
            )

    @property
    def error(self) -> Optional[str]:
        """Returns error message."""
        return self._error

    @property
    def is_running(self) -> bool:
        """Checks if the pipeline is running."""
        return self._is_running


class VideoFilesWriter(ChunkWriter):
    def __init__(
        self,
        base_location: str,
        source_id: str,
        chunk_size: int,
        frame_params: FrameParams,
    ):
        self.base_location = base_location
        self.source_id = source_id
        self.appsrc: Optional[GstApp.AppSrc] = None
        self.pipeline: Optional[Gst.Pipeline] = None
        self.runner: Optional[GstPipelineRunner] = None
        self.frame_params = frame_params
        self.caps = build_caps(frame_params)
        super().__init__(chunk_size)

    def _write_video_frame(
        self,
        frame: VideoFrame,
        data: Optional[bytes],
        frame_num: int,
    ) -> bool:
        if not data:
            return True

        frame_buf: Gst.Buffer = Gst.Buffer.new_wrapped(data)
        frame_buf.pts = convert_ts(frame.pts, frame.time_base)
        frame_buf.dts = (
            convert_ts(frame.dts, frame.time_base)
            if frame.dts is not None
            else Gst.CLOCK_TIME_NONE
        )
        frame_buf.duration = (
            convert_ts(frame.duration, frame.time_base)
            if frame.duration is not None
            else Gst.CLOCK_TIME_NONE
        )
        self.logger.debug(
            'Sending frame with pts=%s to %s',
            frame.pts,
            self.appsrc.get_name(),
        )

        return self.appsrc.push_buffer(frame_buf) == Gst.FlowReturn.OK

    def _write_eos(self, eos: EndOfStream) -> bool:
        return self.appsrc.end_of_stream() == Gst.FlowReturn.OK

    def _open(self):
        self.logger.debug(
            'Creating sink elements for chunk %s of source %s',
            self.chunk_idx,
            self.source_id,
        )
        appsrc_name = 'appsrc'
        filesink_name = 'filesink'
        pipeline = [
            f'appsrc name={appsrc_name} emit-signals=false format=time',
            'queue',
            'adjust_timestamps',
        ]

        if self.frame_params.codec.value.parser is not None:
            pipeline.append(self.frame_params.codec.value.parser)

        if self.frame_params.codec in [
            Codec.H264,
            Codec.HEVC,
            Codec.JPEG,
            Codec.PNG,
        ]:
            pipeline.append(
                'qtmux fragment-duration=1000 fragment-mode=first-moov-then-finalise'
            )
            file_ext = 'mov'
        elif self.frame_params.codec in [
            Codec.VP8,
            Codec.VP9,
        ]:
            pipeline.append('webmmux')
            file_ext = 'webm'
        else:
            self.logger.error(
                'Unsupported codec %s for source %s',
                self.frame_params.codec,
                self.source_id,
            )
            return

        pipeline.append(f'filesink name={filesink_name}')
        self.pipeline: Gst.Pipeline = Gst.parse_launch(' ! '.join(pipeline))
        self.pipeline.set_name(f'video_chunk_{self.source_id}_{self.chunk_idx}')
        self.appsrc: GstApp.AppSrc = self.pipeline.get_by_name(appsrc_name)
        self.appsrc.set_caps(self.caps)

        filesink: Gst.Element = self.pipeline.get_by_name(filesink_name)
        dst_location = self.base_location.replace(
            Patterns.CHUNK_IDX, f'{self.chunk_idx:0{self.chunk_size_digits}}'
        )
        os.makedirs(dst_location, exist_ok=True)
        dst_location = os.path.join(dst_location, f'video.{file_ext}')
        self.logger.info(
            'Writing video from source %s to file %s', self.source_id, dst_location
        )
        filesink.set_property('location', dst_location)
        self.logger.debug(
            'Gst pipeline for chunk %s of source %s has been created',
            self.chunk_idx,
            self.source_id,
        )

        self.runner = GstPipelineRunner(self.pipeline)
        self.runner.startup()
        self.logger.debug(
            'Gst pipeline for chunk %s of source %s has been started',
            self.chunk_idx,
            self.source_id,
        )

    def _close(self):
        self.logger.debug(
            'Stopping and removing sink elements for chunk %s of source %s',
            self.chunk_idx,
            self.source_id,
        )
        self.logger.debug('Sending EOS to %s', self.appsrc.get_name())
        self.appsrc.end_of_stream()
        self.appsrc = None


class VideoFilesSink:
    def __init__(
        self,
        location: str,
        chunk_size: int,
    ):
        self.logger = get_logger(f'{LOGGER_NAME}.{self.__class__.__name__}')
        self.location = location
        self.chunk_size = chunk_size
        self.writers: Dict[str, ChunkWriter] = {}

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

    def _write_video_frame(
        self, video_frame: VideoFrame, content: Optional[bytes]
    ) -> bool:
        frame_params = FrameParams.from_video_frame(video_frame)
        if frame_params.codec not in [
            Codec.H264,
            Codec.HEVC,
            Codec.VP8,
            Codec.VP9,
            Codec.JPEG,
            Codec.PNG,
        ]:
            self.logger.error(
                'Frame %s/%s has unsupported codec %s',
                video_frame.source_id,
                video_frame.pts,
                video_frame.codec,
            )
            return False

        writer = self.writers.get(video_frame.source_id)
        if writer is None:
            src_file_location = get_tag_location(video_frame) or ''
            base_location = get_location(
                self.location, video_frame.source_id, src_file_location
            )
            if self.chunk_size > 0 and Patterns.CHUNK_IDX not in base_location:
                base_location = os.path.join(base_location, Patterns.CHUNK_IDX)

            video_writer = VideoFilesWriter(
                base_location,
                video_frame.source_id,
                self.chunk_size,
                frame_params,
            )
            writer = CompositeChunkWriter(
                [
                    video_writer,
                    MetadataJsonWriter(
                        os.path.join(base_location, 'metadata.json'),
                        self.chunk_size,
                    ),
                ],
                self.chunk_size,
            )
            self.writers[video_frame.source_id] = writer

        return writer.write_video_frame(video_frame, content, video_frame.keyframe)

    def _write_eos(self, eos: EndOfStream):
        self.logger.info('Received EOS from source %s.', eos.source_id)
        writer = self.writers.get(eos.source_id)
        if writer is None:
            return False
        writer.write_eos(eos)
        writer.close()
        return True

    def terminate(self):
        for file_writer in self.writers.values():
            file_writer.close()


def main():
    init_logging()
    # To gracefully shut down the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    logger = get_logger(LOGGER_NAME)
    logger.info(get_starting_message('video files sink adapter'))

    dir_location = req_config('DIR_LOCATION')
    zmq_endpoint = req_config('ZMQ_ENDPOINT')
    zmq_socket_type = opt_config('ZMQ_TYPE', 'SUB')
    zmq_bind = opt_config('ZMQ_BIND', False, strtobool)
    chunk_size = opt_config('CHUNK_SIZE', DEFAULT_CHUNK_SIZE, int)
    source_id = opt_config('SOURCE_ID')
    source_id_prefix = opt_config('SOURCE_ID_PREFIX')

    # possible exceptions will cause app to crash and log error by default
    # no need to handle exceptions here
    source = ZeroMQSource(
        zmq_endpoint,
        zmq_socket_type,
        zmq_bind,
        source_id=source_id,
        source_id_prefix=source_id_prefix,
    )

    video_sink = VideoFilesSink(dir_location, chunk_size)
    Gst.init(None)
    logger.info('Video files sink started')

    try:
        source.start()
        for zmq_message in source:
            video_sink.write(zmq_message)
    except KeyboardInterrupt:
        logger.info('Interrupted')
    finally:
        source.terminate()
        video_sink.terminate()


if __name__ == '__main__':
    main()
