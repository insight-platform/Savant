#!/usr/bin/env python3
import os
import traceback
from distutils.util import strtobool
from typing import Dict, List

from savant_rs.primitives import EndOfStream, VideoFrame
from savant_rs.utils.serialization import Message, load_message_from_bytes

from adapters.python.shared.config import opt_config
from adapters.python.sinks.chunk_writer import ChunkWriter, CompositeChunkWriter
from adapters.python.sinks.metadata_json import (
    MetadataJsonWriter,
    Patterns,
    frame_has_objects,
)
from savant.api.enums import ExternalFrameType
from savant.utils.logging import get_logger, init_logging
from savant.utils.zeromq import ZeroMQSource

LOGGER_NAME = 'adapters.image_files_sink'
DEFAULT_CHUNK_SIZE = 10000


class ImageFilesWriter(ChunkWriter):
    def __init__(self, base_location: str, chunk_size: int):
        self.base_location = base_location
        self.chunk_location = None
        super().__init__(chunk_size)

    def _write_video_frame(self, frame: VideoFrame, data, frame_num: int) -> bool:
        if frame.content.is_external():
            frame_type = ExternalFrameType(frame.content.get_method())
            if frame_type != ExternalFrameType.ZEROMQ:
                self.logger.error('Unsupported frame type "%s".', frame_type.value)
                return False
            if len(data) != 1:
                self.logger.error('Data has %s parts, expected 1.', len(data))
                return False
            content = data[0]
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
        if self.chunk_size > 0:
            self.chunk_location = os.path.join(
                self.base_location,
                f'{self.chunk_idx:04}',
            )
        else:
            self.chunk_location = self.base_location
        self.logger.info('Creating directory %s', self.chunk_location)
        os.makedirs(self.chunk_location, exist_ok=True)


class ImageFilesSink:
    def __init__(
        self,
        location: str,
        chunk_size: int,
        skip_frames_without_objects: bool = False,
    ):
        self.logger = get_logger(f'adapters.{self.__class__.__name__}')
        self.location = location
        self.chunk_size = chunk_size
        self.skip_frames_without_objects = skip_frames_without_objects
        self.writers: Dict[str, ChunkWriter] = {}

    def write(self, message: Message, data: List[bytes]):
        if message.is_video_frame():
            return self._write_video_frame(message.as_video_frame(), data)
        elif message.is_end_of_stream():
            return self._write_eos(message.as_end_of_stream())
        self.logger.debug('Unsupported message type for message %r', message)

    def _write_video_frame(self, video_frame: VideoFrame, data: List[bytes]) -> bool:
        if self.skip_frames_without_objects and not frame_has_objects(video_frame):
            self.logger.debug(
                'Frame %s from source %s does not have objects. Skipping it.',
                video_frame.source_id,
                video_frame.pts,
            )
            return False
        writer = self.writers.get(video_frame.source_id)
        if writer is None:
            base_location = os.path.join(self.location, video_frame.source_id)
            if self.chunk_size > 0:
                json_filename_pattern = f'{Patterns.CHUNK_IDX}.json'
            else:
                json_filename_pattern = 'meta.json'
            writer = CompositeChunkWriter(
                [
                    ImageFilesWriter(base_location, self.chunk_size),
                    MetadataJsonWriter(
                        os.path.join(base_location, json_filename_pattern),
                        self.chunk_size,
                    ),
                ],
                self.chunk_size,
            )
            self.writers[video_frame.source_id] = writer
        return writer.write_video_frame(video_frame, data, video_frame.keyframe)

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
    logger = get_logger(LOGGER_NAME)

    dir_location = os.environ['DIR_LOCATION']
    zmq_endpoint = os.environ['ZMQ_ENDPOINT']
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
            message = load_message_from_bytes(message_bin)
            message.validate_seq_id()
            image_sink.write(message, data)
    except KeyboardInterrupt:
        logger.info('Interrupted')
    finally:
        image_sink.terminate()


if __name__ == '__main__':
    main()
