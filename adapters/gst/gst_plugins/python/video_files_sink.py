import os
from typing import Dict, Optional, Union

from adapters.python.sinks.chunk_writer import ChunkWriter, CompositeChunkWriter
from adapters.python.sinks.metadata_json import MetadataJsonWriter, Patterns
from savant.api import deserialize
from savant.api.enums import ExternalFrameType
from gst_plugins.python.avro_video_demux import build_caps, FrameParams
from savant.gstreamer import GLib, GObject, Gst, GstApp
from savant.gstreamer.codecs import Codec, CODEC_BY_NAME
from savant.gstreamer.utils import LoggerMixin, on_pad_event

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

    def _write(
        self,
        message: Dict,
        data: Union[bytes, Gst.Memory],
        frame_num: Optional[int],
    ) -> bool:
        if 'pts' not in message:
            return True
        frame_pts = message['pts']
        frame_dts = message['dts']
        frame_duration = message['duration']
        if frame_num is None:
            # frame_num should not be None, but we don't use it here anyway
            self.logger.warning('Frame_num is None for frame with PTS %s.', frame_pts)
        if data:
            if isinstance(data, bytes):
                frame_buf: Gst.Buffer = Gst.Buffer.new_wrapped(data)
            else:
                frame_buf: Gst.Buffer = Gst.Buffer.new()
                frame_buf.append_memory(data)
            frame_buf.pts = frame_pts
            frame_buf.dts = Gst.CLOCK_TIME_NONE if frame_dts is None else frame_dts
            frame_buf.duration = (
                Gst.CLOCK_TIME_NONE if frame_duration is None else frame_duration
            )
            self.logger.debug(
                'Sending frame with pts=%s to %s', frame_pts, self.appsrc.get_name()
            )
            if self.appsrc.push_buffer(frame_buf) == Gst.FlowReturn.OK:
                return True
            return False
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
        dst_location = os.path.join(self.base_location, f'{self.chunk_idx:04}.mov')
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
            'Chunk size in frames',
            1,
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
        frame_meta_mapinfo: Gst.MapInfo
        result, frame_meta_mapinfo = buffer.map_range(0, 1, Gst.MapFlags.READ)
        assert result, 'Cannot read buffer.'

        # TODO: Pipeline message types might be extended beyond only VideoFrame
        # Additional checks for audio/raw_tensors/etc. may be required
        schema_name, message = deserialize(frame_meta_mapinfo.data)
        message_with_schema = {**message, 'schema': schema_name}
        if schema_name == 'VideoFrame':
            result = self.handle_video_frame(message_with_schema, buffer)
        elif schema_name == 'EndOfStream':
            result = self.handle_eos(message_with_schema)
        else:
            self.logger.error('Unknown schema "%s"', schema_name)
            result = Gst.FlowReturn.ERROR

        buffer.unmap(frame_meta_mapinfo)
        return result

    def handle_video_frame(self, message: Dict, buffer: Gst.Buffer) -> Gst.FlowReturn:
        source_id = message['source_id']
        frame_params = FrameParams(
            codec=CODEC_BY_NAME[message['codec']],
            width=message['width'],
            height=message['height'],
            framerate=message['framerate'],
        )
        assert frame_params.codec in [Codec.H264, Codec.HEVC, Codec.JPEG, Codec.PNG]
        frame_pts = message['pts']
        frame = message['frame']
        if isinstance(frame, bytes):
            self.logger.debug(
                'Received frame %s from source %s, size: %s bytes',
                frame_pts,
                source_id,
                len(frame) if frame else 0,
            )
        else:
            frame_type = ExternalFrameType(frame['type'])
            if frame_type != ExternalFrameType.ZEROMQ:
                self.logger.error('Unsupported frame type "%s".', frame_type.value)
                return Gst.FlowReturn.ERROR
            if buffer.n_memory() < 2:
                self.logger.error(
                    'Buffer has %s regions of memory, expected at least 2.',
                    buffer.n_memory(),
                )
                return Gst.FlowReturn.ERROR
            frame = buffer.get_memory_range(1, -1)
            self.logger.debug(
                'Received frame %s from source %s, size: %s bytes',
                frame_pts,
                source_id,
                frame.size,
            )

        writer = self.writers.get(source_id)
        if writer is None:
            base_location = os.path.join(self.location, source_id)
            video_writer = VideoFilesWriter(
                base_location,
                source_id,
                self.chunk_size,
                frame_params,
            )
            writer = CompositeChunkWriter(
                [
                    video_writer,
                    MetadataJsonWriter(
                        os.path.join(base_location, f'{Patterns.CHUNK_IDX}.json'),
                        self.chunk_size,
                    ),
                ],
                self.chunk_size,
            )
            self.writers[source_id] = writer
            self.add(video_writer.bin)
            video_writer.bin.sync_state_with_parent()
        if writer.write(message, frame, message['keyframe']):
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR

    def handle_eos(self, message: Dict) -> Gst.FlowReturn:
        source_id = message['source_id']
        self.logger.info('Received EOS from source %s.', source_id)
        writer = self.writers.get(source_id)
        if writer is not None:
            writer.write(message, None, can_start_new_chunk=False, is_frame=False)
            writer.close()
        return Gst.FlowReturn.OK


# register plugin
GObject.type_register(VideoFilesSink)
__gstelementfactory__ = (
    VideoFilesSink.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    VideoFilesSink,
)
