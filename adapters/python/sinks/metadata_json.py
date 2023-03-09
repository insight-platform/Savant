#!/usr/bin/env python3

import json
import logging
import os
import traceback
from distutils.util import strtobool
from typing import Dict, Optional

from adapters.python.sinks.chunk_writer import ChunkWriter
from savant.api import deserialize
from savant.utils.zeromq import ZeroMQSource, build_topic_prefix


class Patterns:
    SOURCE_ID = '%source_id'
    SRC_FILENAME = '%src_filename'
    CHUNK_IDX = '%chunk_idx'


class MetadataJsonWriter(ChunkWriter):
    def __init__(self, pattern: str, chunk_size: int):
        self.pattern = pattern
        super().__init__(chunk_size)

    def _write(self, metadata: Dict, frame_num: Optional[int]) -> bool:
        self.logger.debug('Writing meta to file %s', self.location)
        data = {k: v for k, v in metadata.items() if k != 'frame'}
        if frame_num is not None:
            data['frame_num'] = frame_num
        try:
            json.dump(data, self.file)
            self.file.write('\n')
        except Exception:
            traceback.print_exc()
            return False
        return True

    def _flush(self):
        self.logger.info('Flushing file %s', self.file.name)
        self.file.flush()

    def _close(self):
        self.logger.info('Closing file %s', self.file.name)
        self.file.close()

    def _open(self):
        if self.chunk_size > 0:
            self.location = self.pattern.replace(
                Patterns.CHUNK_IDX, f'{self.chunk_idx:04}'
            )
        else:
            self.location = self.pattern
        self.lines = 0
        self.logger.info('Opening file %s', self.location)
        os.makedirs(os.path.dirname(self.location), exist_ok=True)
        self.file = open(self.location, 'w')


class MetadataJsonSink:
    """Writes frames metadata to JSON files."""

    def __init__(
        self,
        location: str,
        skip_frames_without_objects: bool = True,
        chunk_size: int = 0,
    ):
        self.logger = logging.getLogger(__name__)
        self.skip_frames_without_objects = skip_frames_without_objects
        self.chunk_size = chunk_size
        self.writers: Dict[str, MetadataJsonWriter] = {}
        self.last_writer_per_source: Dict[str, MetadataJsonWriter] = {}

        path, ext = os.path.splitext(location)
        ext = ext or '.json'
        if self.chunk_size > 0 and Patterns.CHUNK_IDX not in location:
            path += f'_{Patterns.CHUNK_IDX}'
        self.location = f'{path}{ext}'

    def terminate(self):
        for file_writer in self.writers.values():
            file_writer.close()

    def write(self, schema_name: str, message: Dict):
        message_with_schema = {**message, 'schema': schema_name}
        if schema_name == 'VideoFrame':
            return self._write_video_frame(message_with_schema)
        elif schema_name == 'EndOfStream':
            return self._write_eos(message_with_schema)
        self.logger.error('Unknown schema "%s"', schema_name)

    def _write_video_frame(self, message: Dict):
        source_id = message['source_id']
        pts = message['pts']
        if self.skip_frames_without_objects and not frame_has_objects(message):
            self.logger.debug(
                'Frame %s from source %s does not have objects. Skipping it.',
                source_id,
                pts,
            )
            return False

        src_file_location = message.get('tags', {}).get('location') or ''
        location = self.get_location(source_id, src_file_location)
        writer = self.writers.get(location)
        last_writer = self.last_writer_per_source.get(source_id)
        if writer is None:
            writer = MetadataJsonWriter(location, self.chunk_size)
            self.writers[location] = writer
        if writer is not last_writer:
            if last_writer is not None:
                last_writer.flush()
            self.last_writer_per_source[source_id] = writer
        return writer.write(message, message['keyframe'])

    def _write_eos(self, message: Dict):
        source_id = message['source_id']
        self.logger.info('Received EOS from source %s.', source_id)
        writer = self.last_writer_per_source.get(source_id)
        if writer is None:
            return False
        result = writer.write(message, can_start_new_chunk=False, is_frame=False)
        writer.flush()
        return result

    def get_location(
        self,
        source_id: str,
        src_file_location: str,
    ):
        location = self.location.replace(Patterns.SOURCE_ID, source_id)
        src_filename = os.path.splitext(os.path.basename(src_file_location))[0]
        location = location.replace(Patterns.SRC_FILENAME, src_filename)
        return location


def frame_has_objects(message: Dict):
    metadata = message.get('metadata')
    if not metadata:
        return False
    return bool(metadata.get('objects'))


def main():
    logging.basicConfig(
        level=os.environ.get('LOGLEVEL', 'INFO'),
        format='%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] %(message)s',
    )
    location = os.environ['LOCATION']
    zmq_endpoint = os.environ['ZMQ_ENDPOINT']
    zmq_socket_type = os.environ.get('ZMQ_TYPE', 'SUB')
    zmq_bind = strtobool(os.environ.get('ZMQ_BIND', 'false'))
    skip_frames_without_objects = strtobool(
        os.environ.get('SKIP_FRAMES_WITHOUT_OBJECTS', 'false')
    )
    chunk_size = int(os.environ.get('CHUNK_SIZE', 0))
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

    sink = MetadataJsonSink(location, skip_frames_without_objects, chunk_size)
    logging.info('Metadata JSON sink started')

    try:
        for message_bin in source:
            schema_name, message = deserialize(message_bin)
            sink.write(schema_name, message)
    except KeyboardInterrupt:
        logging.info('Interrupted')
    finally:
        sink.terminate()


if __name__ == '__main__':
    main()
