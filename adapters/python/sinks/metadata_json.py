#!/usr/bin/env python3

import json
import logging
import os
import traceback
from distutils.util import strtobool
from typing import Any, Dict, Optional

from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    AttributeValueType,
    EndOfStream,
    VideoFrame,
)
from savant_rs.utils.serialization import Message, load_message_from_bytes
from savant_rs.video_object_query import MatchQuery

from adapters.python.sinks.chunk_writer import ChunkWriter
from savant.api.constants import DEFAULT_NAMESPACE
from savant.api.parser import parse_video_frame
from savant.utils.logging import init_logging
from savant.utils.zeromq import ZeroMQSource, build_topic_prefix

LOGGER_NAME = 'savant.adapters.metadata_json_sink'


class Patterns:
    SOURCE_ID = '%source_id'
    SRC_FILENAME = '%src_filename'
    CHUNK_IDX = '%chunk_idx'


class MetadataJsonWriter(ChunkWriter):
    def __init__(self, pattern: str, chunk_size: int):
        self.pattern = pattern
        super().__init__(chunk_size)

    def _write_video_frame(self, frame: VideoFrame, data: Any, frame_num: int) -> bool:
        metadata = parse_video_frame(frame)
        metadata['schema'] = 'VideoFrame'
        return self._write_meta_to_file(metadata, frame_num)

    def _write_eos(self, eos: EndOfStream) -> bool:
        metadata = {'source_id': eos.source_id, 'schema': 'EndOfStream'}
        return self._write_meta_to_file(metadata, None)

    def _write_meta_to_file(
        self,
        metadata: Dict,
        frame_num: Optional[int],
    ) -> bool:
        self.logger.debug('Writing meta to file %s', self.location)
        if frame_num is not None:
            metadata['frame_num'] = frame_num
        try:
            json.dump(metadata, self.file)
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
        self.logger = logging.getLogger(f'savant.adapters.{self.__class__.__name__}')
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

    def write(self, message: Message):
        if message.is_video_frame():
            return self._write_video_frame(message.as_video_frame())
        elif message.is_end_of_stream():
            return self._write_eos(message.as_end_of_stream())
        self.logger.debug('Unsupported message type for message %r', message)

    def _write_video_frame(self, frame: VideoFrame):
        if self.skip_frames_without_objects and not frame_has_objects(frame):
            self.logger.debug(
                'Frame %s from source %s does not have objects. Skipping it.',
                frame.source_id,
                frame.pts,
            )
            return False

        src_file_location = get_tag_location(frame) or ''
        location = self.get_location(frame.source_id, src_file_location)
        writer = self.writers.get(location)
        last_writer = self.last_writer_per_source.get(frame.source_id)
        if writer is None:
            writer = MetadataJsonWriter(location, self.chunk_size)
            self.writers[location] = writer
        if writer is not last_writer:
            if last_writer is not None:
                last_writer.flush()
            self.last_writer_per_source[frame.source_id] = writer

        return writer.write_video_frame(frame, None, frame.keyframe)

    def _write_eos(self, eos: EndOfStream):
        self.logger.info('Received EOS from source %s.', eos.source_id)
        writer = self.last_writer_per_source.get(eos.source_id)
        if writer is None:
            return False
        result = writer.write_eos(eos)
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


def frame_has_objects(frame: VideoFrame):
    return bool(frame.access_objects(MatchQuery.idle()))


def get_tag_location(frame: VideoFrame):
    attr: Optional[Attribute] = frame.get_attribute(DEFAULT_NAMESPACE, 'location')
    if attr is None or not attr.values:
        return None
    value: AttributeValue = attr.values[0]
    if value.value_type != AttributeValueType.String:
        return None
    return value.as_string()


def main():
    init_logging()
    logger = logging.getLogger(LOGGER_NAME)
    location = os.environ['LOCATION']
    zmq_endpoint = os.environ['ZMQ_ENDPOINT']
    zmq_socket_type = os.environ.get('ZMQ_TYPE', 'SUB')
    zmq_bind = bool(strtobool(os.environ.get('ZMQ_BIND', 'false')))
    skip_frames_without_objects = bool(
        strtobool(os.environ.get('SKIP_FRAMES_WITHOUT_OBJECTS', 'false'))
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
    logger.info('Metadata JSON sink started')

    try:
        source.start()
        for message_bin, *data in source:
            message = load_message_from_bytes(message_bin)
            sink.write(message)
    except KeyboardInterrupt:
        logger.info('Interrupted')
    finally:
        sink.terminate()


if __name__ == '__main__':
    main()
