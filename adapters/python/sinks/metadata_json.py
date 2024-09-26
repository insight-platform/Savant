#!/usr/bin/env python3

import json
import os
import signal
import traceback
from typing import Dict, Optional

from savant_rs.match_query import MatchQuery
from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    AttributeValueType,
    EndOfStream,
    VideoFrame,
)

from adapters.python.sinks.chunk_writer import ChunkWriter
from savant.api.constants import DEFAULT_NAMESPACE
from savant.api.parser import parse_video_frame
from savant.utils.config import opt_config, req_config, strtobool
from savant.utils.logging import get_logger, init_logging
from savant.utils.welcome import get_starting_message
from savant.utils.zeromq import ZeroMQMessage, ZeroMQSource

LOGGER_NAME = 'adapters.metadata_json_sink'


class Patterns:
    SOURCE_ID = '%source_id'
    SRC_FILENAME = '%src_filename'
    CHUNK_IDX = '%chunk_idx'


class MetadataJsonWriter(ChunkWriter):
    def __init__(self, pattern: str, chunk_size: int):
        super().__init__(chunk_size, logger_prefix=LOGGER_NAME)
        self.pattern = pattern
        self.logger.info('File name pattern is %s', self.pattern)

    def _write_video_frame(
        self,
        frame: VideoFrame,
        content: Optional[bytes],
        frame_num: int,
    ) -> bool:
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
        self.location = self.pattern.replace(
            Patterns.CHUNK_IDX, f'{self.chunk_idx:0{self.chunk_size_digits}}'
        )
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
        self.logger = get_logger(f'{LOGGER_NAME}.{self.__class__.__name__}')
        self.skip_frames_without_objects = skip_frames_without_objects
        self.chunk_size = chunk_size
        self.writers: Dict[str, MetadataJsonWriter] = {}
        self.last_writer_per_source: Dict[str, (str, MetadataJsonWriter)] = {}

        path, ext = os.path.splitext(location)
        ext = ext or '.json'
        if self.chunk_size > 0 and Patterns.CHUNK_IDX not in location:
            path += f'_{Patterns.CHUNK_IDX}'
        self.location = f'{path}{ext}'

    def terminate(self):
        for file_writer in self.writers.values():
            file_writer.close()

    def write(self, zmq_message: ZeroMQMessage):
        message = zmq_message.message
        message.validate_seq_id()
        if message.is_video_frame():
            return self._write_video_frame(message.as_video_frame())
        elif message.is_end_of_stream():
            return self._write_eos(message.as_end_of_stream())
        self.logger.debug('Unsupported message type for message %r', message)

    def _write_video_frame(self, video_frame: VideoFrame):
        if self.skip_frames_without_objects and not frame_has_objects(video_frame):
            self.logger.debug(
                'Frame %s from source %s does not have objects. Skipping it.',
                video_frame.source_id,
                video_frame.pts,
            )
            return False

        src_file_location = get_tag_location(video_frame) or 'unknown'
        location = get_location(self.location, video_frame.source_id, src_file_location)
        writer = self.writers.get(location)
        last_source_location, last_source_writer = self.last_writer_per_source.get(
            video_frame.source_id
        ) or (None, None)

        if writer is None:
            writer = MetadataJsonWriter(location, self.chunk_size)
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

        return writer.write_video_frame(video_frame, None, video_frame.keyframe)

    def _write_eos(self, eos: EndOfStream):
        self.logger.info('Received EOS from source %s.', eos.source_id)
        writer = self.last_writer_per_source.get(eos.source_id)
        if writer is None:
            return False
        result = writer.write_eos(eos)
        writer.flush()
        return result


def get_location(
    location_pattern,
    source_id: str,
    src_file_location: str,
):
    src_filename = os.path.splitext(os.path.basename(src_file_location))[0]
    location = location_pattern.replace(Patterns.SOURCE_ID, source_id).replace(
        Patterns.SRC_FILENAME, src_filename
    )
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
    # To gracefully shut down the adapter on SIGTERM (raise KeyboardInterrupt)
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))

    logger = get_logger(LOGGER_NAME)
    logger.info(get_starting_message('metadata sink adapter'))

    location = req_config('FILENAME_PATTERN')
    zmq_endpoint = req_config('ZMQ_ENDPOINT')
    zmq_socket_type = opt_config('ZMQ_TYPE', 'SUB')
    zmq_bind = opt_config('ZMQ_BIND', False, strtobool)
    skip_frames_without_objects = opt_config(
        'SKIP_FRAMES_WITHOUT_OBJECTS', False, strtobool
    )
    chunk_size = opt_config('CHUNK_SIZE', 0, int)
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

    sink = MetadataJsonSink(location, skip_frames_without_objects, chunk_size)
    logger.info('Metadata JSON sink started')

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
