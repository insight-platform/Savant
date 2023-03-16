#!/usr/bin/env python3

import logging
import os
import traceback
from distutils.util import strtobool
from typing import Dict, List, Optional

from adapters.python.sinks.chunk_writer import ChunkWriter, CompositeChunkWriter
from adapters.python.sinks.metadata_json import (
    frame_has_objects,
    MetadataJsonWriter,
    Patterns,
)
from savant.api import deserialize
from savant.api.enums import ExternalFrameType
from savant.utils.zeromq import ZeroMQSource, build_topic_prefix

DEFAULT_CHUNK_SIZE = 10000


class ImageFilesWriter(ChunkWriter):
    def __init__(self, base_location: str, chunk_size: int):
        self.base_location = base_location
        self.chunk_location = None
        super().__init__(chunk_size)

    def _write(
        self,
        message: Dict,
        data: List[bytes],
        frame_num: Optional[int],
    ) -> bool:
        frame = message.get('frame')
        if frame is None:
            return True
        if frame_num is None:
            # Cannot get file location for frame_num=None
            self.logger.warning(
                'Got frame of size %s with PTS %s, but "frame_num" is None. Not saving it to a file.',
                len(frame),
                message.get('pts'),
            )
            return True
        if isinstance(frame, dict):
            frame_type = ExternalFrameType(frame['type'])
            if frame_type != ExternalFrameType.ZEROMQ:
                self.logger.error('Unsupported frame type "%s".', frame_type.value)
                return False
            if len(data) != 1:
                self.logger.error('Data has %s parts, expected 1.', len(data))
                return False
            frame = data[0]
        codec = message['codec']
        filepath = os.path.join(
            self.chunk_location, f'{frame_num:0{self.chunk_size_digits}}.{codec}'
        )
        self.logger.debug('Writing frame to file %s', filepath)
        try:
            with open(filepath, 'wb') as f:
                f.write(frame)
        except Exception:
            traceback.print_exc()
            return False
        return True

    def _open(self):
        self.chunk_location = os.path.join(self.base_location, f'{self.chunk_idx:04}')
        self.logger.info('Creating directory %s', self.chunk_location)
        os.makedirs(self.chunk_location, exist_ok=True)


class ImageFilesSink:
    def __init__(
        self,
        location: str,
        chunk_size: int,
        skip_frames_without_objects: bool = False,
    ):
        self.logger = logging.getLogger(__name__)
        self.location = location
        self.chunk_size = chunk_size
        self.skip_frames_without_objects = skip_frames_without_objects
        self.writers: Dict[str, ChunkWriter] = {}

    def write(self, schema_name: str, message: Dict, data: List[bytes]):
        message_with_schema = {**message, 'schema': schema_name}
        if schema_name == 'VideoFrame':
            return self._write_video_frame(message_with_schema, data)
        elif schema_name == 'EndOfStream':
            return self._write_eos(message_with_schema)
        self.logger.error('Unknown schema "%s"', schema_name)

    def _write_video_frame(self, message: Dict, data: List[bytes]) -> bool:
        source_id = message['source_id']
        pts = message['pts']
        if self.skip_frames_without_objects and not frame_has_objects(message):
            self.logger.debug(
                'Frame %s from source %s does not have objects. Skipping it.',
                source_id,
                pts,
            )
            return False
        writer = self.writers.get(source_id)
        if writer is None:
            base_location = os.path.join(self.location, source_id)
            writer = CompositeChunkWriter(
                [
                    ImageFilesWriter(base_location, self.chunk_size),
                    MetadataJsonWriter(
                        os.path.join(base_location, f'{Patterns.CHUNK_IDX}.json'),
                        self.chunk_size,
                    ),
                ],
                self.chunk_size,
            )
            self.writers[source_id] = writer
        return writer.write(message, data, message.get('keyframe'))

    def _write_eos(self, message: Dict):
        source_id = message['source_id']
        self.logger.info('Received EOS from source %s.', source_id)
        writer = self.writers.get(source_id)
        if writer is None:
            return False
        writer.write(message, None, can_start_new_chunk=False, is_frame=False)
        writer.flush()
        return True

    def terminate(self):
        for file_writer in self.writers.values():
            file_writer.close()


def main():
    logging.basicConfig(
        level=os.environ.get('LOGLEVEL', 'INFO'),
        format='%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] %(message)s',
    )
    dir_location = os.environ['DIR_LOCATION']
    zmq_endpoint = os.environ['ZMQ_ENDPOINT']
    zmq_socket_type = os.environ.get('ZMQ_TYPE', 'SUB')
    zmq_bind = strtobool(os.environ.get('ZMQ_BIND', 'false'))
    skip_frames_without_objects = strtobool(
        os.environ.get('SKIP_FRAMES_WITHOUT_OBJECTS', 'false')
    )
    chunk_size = int(os.environ.get('CHUNK_SIZE', DEFAULT_CHUNK_SIZE))
    topic_prefix = build_topic_prefix(
        source_id=os.environ.get('SOURCE_ID'),
        source_id_prefix=os.environ.get('SOURCE_ID_PREFIX'),
    )

    # possible exceptions will cause app to crash and log error by default
    # no need to handle exceptions here
    source = ZeroMQSource(
        zmq_endpoint,
        zmq_socket_type,
        zmq_bind,
        topic_prefix=topic_prefix,
    )

    image_sink = ImageFilesSink(dir_location, chunk_size, skip_frames_without_objects)
    logging.info('Image files sink started')

    try:
        for message_bin, *data in source:
            schema_name, message = deserialize(message_bin)
            image_sink.write(schema_name, message, data)
    except KeyboardInterrupt:
        logging.info('Interrupted')
    finally:
        image_sink.terminate()


if __name__ == '__main__':
    main()
