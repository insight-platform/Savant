#!/usr/bin/env python3
import os
import signal
import traceback
from typing import Dict, Optional

from savant_rs.primitives import EndOfStream, VideoFrame

from adapters.python.sinks.chunk_writer import ChunkWriter, CompositeChunkWriter
from adapters.python.sinks.metadata_json import (
    MetadataJsonSink,
    MetadataJsonWriter,
    Patterns,
    frame_has_objects,
    get_location,
    get_tag_location,
)
from savant.api.enums import ExternalFrameType
from savant.utils.config import opt_config, req_config, strtobool
from savant.utils.logging import get_logger, init_logging
from savant.utils.welcome import get_starting_message
from savant.utils.zeromq import ZeroMQMessage, ZeroMQSource

LOGGER_NAME = 'adapters.image_files_sink'
DEFAULT_CHUNK_SIZE = 10000


class ImageFilesWriter(ChunkWriter):
    def __init__(self, base_location: str, chunk_size: int):
        self.base_location = base_location
        self.chunk_location = None
        super().__init__(chunk_size, logger_prefix=LOGGER_NAME)

    def _write_video_frame(
        self,
        frame: VideoFrame,
        content: Optional[bytes],
        frame_num: int,
    ) -> bool:
        if frame.content.is_external():
            frame_type = ExternalFrameType(frame.content.get_method())
            if frame_type != ExternalFrameType.ZEROMQ:
                self.logger.error('Unsupported frame type "%s".', frame_type.value)
                return False
            if not content:
                self.logger.error(
                    'Frame %s/%s has no content data', frame.source_id, frame.pts
                )
                return False
        elif frame.content.is_internal():
            content = frame.content.get_data_as_bytes()
        else:
            return True

        filepath = os.path.join(
            self.chunk_location,
            f'{frame_num:0{self.chunk_size_digits}}.{frame.codec}',
        )
        self.logger.debug('Writing frame to file %s', filepath)
        try:
            with open(filepath, 'wb') as f:
                f.write(content)
        except Exception:
            traceback.print_exc()
            return False

        return True

    def _write_eos(self, eos: EndOfStream) -> bool:
        return True

    def _open(self):
        self.chunk_location = self.base_location.replace(
            Patterns.CHUNK_IDX, f'{self.chunk_idx:0{self.chunk_size_digits}}'
        )
        self.logger.info('Creating directory %s', self.chunk_location)
        os.makedirs(self.chunk_location, exist_ok=True)


class ImageFilesSink:
    def __init__(
        self,
        location: str,
        chunk_size: int,
        skip_frames_without_objects: bool = False,
    ):
        self.logger = get_logger(f'{LOGGER_NAME}.{self.__class__.__name__}')
        self.location = location
        self.chunk_size = chunk_size
        self.skip_frames_without_objects = skip_frames_without_objects
        self.writers: Dict[str, ChunkWriter] = {}
        self.last_writer_per_source: Dict[str, (str, ChunkWriter)] = {}

    def write(self, zmq_message: ZeroMQMessage):
        message = zmq_message.message
        message.validate_seq_id()
        if message.is_video_frame():
            return self._write_video_frame(
                message.as_video_frame(),
                zmq_message.content,
            )
        elif message.is_end_of_stream():
            return self._write_eos(message.as_end_of_stream())
        self.logger.debug('Unsupported message type for message %r', message)

    def _write_video_frame(self, video_frame: VideoFrame, content: bytes) -> bool:
        if self.skip_frames_without_objects and not frame_has_objects(video_frame):
            self.logger.debug(
                'Frame %s from source %s does not have objects. Skipping it.',
                video_frame.source_id,
                video_frame.pts,
            )
            return False
        src_file_location = get_tag_location(video_frame) or 'unknown'
        location = get_location(self.location, video_frame.source_id, src_file_location)
        if self.chunk_size > 0 and Patterns.CHUNK_IDX not in location:
            location = os.path.join(location, Patterns.CHUNK_IDX)
        writer = self.writers.get(location)
        last_source_location, last_source_writer = self.last_writer_per_source.get(
            video_frame.source_id
        ) or (None, None)

        if writer is None:
            writer = CompositeChunkWriter(
                [
                    ImageFilesWriter(os.path.join(location, 'images'), self.chunk_size),
                    MetadataJsonWriter(
                        os.path.join(location, 'metadata.json'),
                        self.chunk_size,
                    ),
                ],
                self.chunk_size,
            )
            self.writers[location] = writer
            if writer is not last_source_writer:
                if last_source_writer is not None:
                    self.logger.info(
                        'Flushing previous writer for source=%s, location=%s',
                        video_frame.source_id,
                        last_source_location,
                    )
                    last_source_writer.flush()
                    self.logger.info(
                        'Removing previous writer for source=%s, location=%s',
                        video_frame.source_id,
                        last_source_location,
                    )
                    del self.writers[last_source_location]
                self.logger.info(
                    'New writer for source=%s, location=%s is initialized, amount of resident writers is %d',
                    video_frame.source_id,
                    location,
                    len(self.writers),
                )
                self.last_writer_per_source[video_frame.source_id] = (location, writer)

        return writer.write_video_frame(video_frame, content, video_frame.keyframe)

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
    # To gracefully shut down the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    logger = get_logger(LOGGER_NAME)
    logger.info(get_starting_message('image files sink adapter'))

    dir_location = req_config('DIR_LOCATION')
    zmq_endpoint = req_config('ZMQ_ENDPOINT')
    zmq_socket_type = opt_config('ZMQ_TYPE', 'SUB')
    zmq_bind = opt_config('ZMQ_BIND', False, strtobool)
    skip_frames_without_objects = opt_config(
        'SKIP_FRAMES_WITHOUT_OBJECTS', False, strtobool
    )
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

    image_sink = ImageFilesSink(dir_location, chunk_size, skip_frames_without_objects)
    logger.info('Image files sink started')

    try:
        source.start()
        for zmq_message in source:
            image_sink.write(zmq_message)
    except KeyboardInterrupt:
        logger.info('Interrupted')
    finally:
        source.terminate()
        image_sink.terminate()


if __name__ == '__main__':
    main()
