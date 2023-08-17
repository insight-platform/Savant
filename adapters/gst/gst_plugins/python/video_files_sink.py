import os
from typing import Dict, Optional, Union

from savant_rs.primitives import EndOfStream, VideoFrame

from adapters.python.sinks.chunk_writer import (ChunkWriter,
                                                CompositeChunkWriter)
from adapters.python.sinks.metadata_json import MetadataJsonWriter, Patterns
from gst_plugins.python.savant_rs_video_demux import FrameParams, build_caps
from savant.api.enums import ExternalFrameType
from savant.api.parser import convert_ts
from savant.gstreamer import GLib, GObject, Gst, GstApp
from savant.gstreamer.codecs import Codec
from savant.gstreamer.utils import load_message_from_gst_buffer, on_pad_event
from savant.utils.logging import LoggerMixin

DEFAULT_CHUNK_SIZE = 10000


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
        self.bin: Gst.Bin = Gst.Bin.new(f'sink_bin_{source_id}')
        self.frame_params = frame_params
        self.caps = build_caps(frame_params)
        super().__init__(chunk_size)

    def _write_video_frame(
        self,
        frame: VideoFrame,
        data: Optional[Union[bytes, Gst.Memory]],
        frame_num: int,
    ) -> bool:
        if not data:
            return True

        if isinstance(data, bytes):
            frame_buf: Gst.Buffer = Gst.Buffer.new_wrapped(data)
        else:
            frame_buf: Gst.Buffer = Gst.Buffer.new()
            frame_buf.append_memory(data)

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
        return True

    def _open(self):
        self.logger.debug(
            'Creating sink elements for chunk %s of source %s',
            self.chunk_idx,
            self.source_id,
        )
        appsrc_name = f'appsrc_{self.source_id}_{self.chunk_idx}'
        filesink_name = f'filesink_{self.source_id}_{self.chunk_idx}'
        sink: Gst.Bin = Gst.parse_bin_from_description(
            ' ! '.join(
                [
                    f'appsrc name={appsrc_name} emit-signals=false format=time',
                    'queue',
                    'adjust_timestamps',
                    self.frame_params.codec.value.parser,
                    'qtmux fragment-duration=1000 fragment-mode=first-moov-then-finalise',
                    f'filesink name={filesink_name}',
                ]
            ),
            False,
        )
        sink.set_name(f'sink_bin_{self.source_id}_{self.chunk_idx}')
        self.appsrc: GstApp.AppSrc = sink.get_by_name(appsrc_name)
        self.appsrc.set_caps(self.caps)

        filesink: Gst.Element = sink.get_by_name(filesink_name)
        os.makedirs(self.base_location, exist_ok=True)
        if self.chunk_size > 0:
            dst_location = os.path.join(self.base_location, f'{self.chunk_idx:04}.mov')
        else:
            dst_location = os.path.join(self.base_location, f'video.mov')
        self.logger.info(
            'Writing video from source %s to file %s', self.source_id, dst_location
        )
        filesink.set_property('location', dst_location)

        self.bin.add(sink)

        filesink.get_static_pad('sink').add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.EOS: self.on_sink_pad_eos},
            sink,
            self.chunk_idx,
        )
        sink.sync_state_with_parent()
        self.logger.debug(
            'Sink elements for chunk %s of source %s created',
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

    def on_sink_pad_eos(
        self, pad: Gst.Pad, event: Gst.Event, sink: Gst.Element, chunk_idx: int
    ):
        self.logger.debug(
            'Got EOS from pad %s of %s', pad.get_name(), pad.parent.get_name()
        )
        GLib.idle_add(self._remove_branch, sink, chunk_idx)
        return Gst.PadProbeReturn.HANDLED

    def _remove_branch(self, sink: Gst.Element, chunk_idx: int):
        self.logger.debug('Removing element %s', sink.get_name())
        sink.set_locked_state(True)
        sink.set_state(Gst.State.NULL)
        self.bin.remove(sink)
        self.logger.debug(
            'Sink elements for chunk %s of source %s removed', chunk_idx, self.source_id
        )

        return False


class VideoFilesSink(LoggerMixin, Gst.Bin):
    """Writes frames as video files."""

    GST_PLUGIN_NAME = 'video_files_sink'

    __gstmetadata__ = (
        'Video files sink',
        'Bin/Sink/File',
        'Writes frames as video files',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'sink',
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        ),
    )

    __gproperties__ = {
        'location': (
            GObject.TYPE_STRING,
            'Output directory location',
            'Location of the directory for output files',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'chunk-size': (
            int,
            'Chunk size',
            'Chunk size in frames (0 to disable chunks).',
            0,
            GObject.G_MAXINT,
            DEFAULT_CHUNK_SIZE,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        self.writers: Dict[str, ChunkWriter] = {}

        self.location: str = None
        self.chunk_size: int = DEFAULT_CHUNK_SIZE

        self.sink_pad: Gst.Pad = Gst.Pad.new_from_template(
            Gst.PadTemplate.new(
                'sink',
                Gst.PadDirection.SINK,
                Gst.PadPresence.ALWAYS,
                Gst.Caps.new_any(),
            ),
            'sink',
        )
        self.sink_pad.set_chain_function(self.handle_buffer)
        assert self.add_pad(self.sink_pad), 'Failed to add sink pad.'

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates
            the metadata required to specify parameters
        """
        if prop.name == 'location':
            return self.location
        if prop.name == 'chunk-size':
            return self.chunk_size
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: structure that encapsulates
            the metadata required to specify parameters
        :param value: new value for param, type dependents on param
        """
        if prop.name == 'location':
            self.location = value
        elif prop.name == 'chunk-size':
            self.chunk_size = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_state(self, state: Gst.State):
        self.logger.info('Changing state from %s to %s', self.current_state, state)
        if self.current_state == Gst.State.NULL and state != Gst.State.NULL:
            assert self.location is not None, '"location" property is required'
        return Gst.Bin.do_set_state(self, state)

    def handle_buffer(self, sink_pad: Gst.Pad, buffer: Gst.Buffer) -> Gst.FlowReturn:
        self.logger.debug(
            'Handling buffer of size %s with timestamp %s',
            buffer.get_size(),
            buffer.pts,
        )

        message = load_message_from_gst_buffer(buffer)
        # TODO: Pipeline message types might be extended beyond only VideoFrame
        # Additional checks for audio/raw_tensors/etc. may be required
        if message.is_video_frame():
            result = self.handle_video_frame(message.as_video_frame(), buffer)
        elif message.is_end_of_stream():
            result = self.handle_eos(message.as_end_of_stream())
        else:
            self.logger.debug('Unsupported message type for message %r', message)
            result = Gst.FlowReturn.OK

        return result

    def handle_video_frame(
        self,
        frame: VideoFrame,
        buffer: Gst.Buffer,
    ) -> Gst.FlowReturn:
        frame_params = FrameParams.from_video_frame(frame)
        assert frame_params.codec in [
            Codec.H264,
            Codec.HEVC,
            Codec.JPEG,
            Codec.PNG,
        ], f'Unsupported codec {frame.codec}'
        if frame.content.is_none():
            self.logger.debug(
                'Received frame %s from source %s is empty',
                frame.pts,
                frame.source_id,
            )
            content = None
        elif frame.content.is_internal():
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
                return Gst.FlowReturn.ERROR
            if buffer.n_memory() < 2:
                self.logger.error(
                    'Buffer has %s regions of memory, expected at least 2.',
                    buffer.n_memory(),
                )
                return Gst.FlowReturn.ERROR

            content = buffer.get_memory_range(1, -1)
            self.logger.debug(
                'Received frame %s from source %s, size: %s bytes',
                frame.pts,
                frame.source_id,
                content.size,
            )

        writer = self.writers.get(frame.source_id)
        if writer is None:
            base_location = os.path.join(self.location, frame.source_id)
            if self.chunk_size > 0:
                json_filename_pattern = f'{Patterns.CHUNK_IDX}.json'
            else:
                json_filename_pattern = 'meta.json'
            video_writer = VideoFilesWriter(
                base_location,
                frame.source_id,
                self.chunk_size,
                frame_params,
            )
            writer = CompositeChunkWriter(
                [
                    video_writer,
                    MetadataJsonWriter(
                        os.path.join(base_location, json_filename_pattern),
                        self.chunk_size,
                    ),
                ],
                self.chunk_size,
            )
            self.writers[frame.source_id] = writer
            self.add(video_writer.bin)
            video_writer.bin.sync_state_with_parent()

        if writer.write_video_frame(frame, content, frame.keyframe):
            return Gst.FlowReturn.OK

        return Gst.FlowReturn.ERROR

    def handle_eos(self, eos: EndOfStream) -> Gst.FlowReturn:
        self.logger.info('Received EOS from source %s.', eos.source_id)
        writer = self.writers.get(eos.source_id)
        if writer is not None:
            writer.write_eos(eos)
            writer.close()
        return Gst.FlowReturn.OK


# register plugin
GObject.type_register(VideoFilesSink)
__gstelementfactory__ = (
    VideoFilesSink.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    VideoFilesSink,
)
