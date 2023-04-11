import re
import subprocess
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO, List, Optional, Union
from urllib.parse import urlparse

from savant.gstreamer import GLib, GObject, Gst
from savant.gstreamer.codecs import Codec, CODEC_BY_CAPS_NAME
from savant.gstreamer.metadata import DEFAULT_FRAMERATE
from savant.gstreamer.utils import LoggerMixin, on_pad_event


class FileType(Enum):
    VIDEO = 'video'
    PICTURE = 'picture'


MIME_TYPE_REGEX = {
    FileType.VIDEO: re.compile(r'video/.*'),
    FileType.PICTURE: re.compile(r'image/(jpeg|png)'),
}


class MediaFilesSrcBin(LoggerMixin, Gst.Bin):
    GST_PLUGIN_NAME = 'media_files_src_bin'

    __gstmetadata__ = (
        'Media files source bin',
        'Bin',
        'Reads media file or all media files in directory',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = Gst.PadTemplate.new(
        'src',
        Gst.PadDirection.SRC,
        Gst.PadPresence.ALWAYS,
        Gst.Caps.from_string(';'.join(x.value.caps_with_params for x in Codec)),
    )

    __gproperties__ = {
        'file-type': (
            GObject.TYPE_STRING,
            'File type',
            'Type of media files to read ("video" or "picture")',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'location': (
            GObject.TYPE_STRING,
            'Location',
            'Location of media file or directory with media files to read',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        # TODO: make fraction
        'framerate': (
            str,
            'Framerate',
            'Framerate for pictures. Used only when file-type=picture.',
            DEFAULT_FRAMERATE,
            GObject.ParamFlags.READWRITE,
        ),
        'sort-by-time': (
            bool,
            'Sort files by modification time',
            'Sort files by modification time. Otherwise files will be sorted by name.',
            False,
            GObject.ParamFlags.READWRITE,
        ),
        'loop-file': (
            bool,
            'Loop single video file',
            'Loop single video file',
            False,
            GObject.ParamFlags.READWRITE,
        ),
        'download-path': (
            str,
            'Path to download files from remote storage',
            'Path to download files from remote storage',
            None,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # properties
        self.file_type: FileType = None
        self.location: Optional[Union[Path, str]] = None
        self.framerate = Fraction(DEFAULT_FRAMERATE)
        self.frame_duration = int(Gst.SECOND / self.framerate)
        self.sort_by_time = False
        self.loop_file = False
        self.download_path: Optional[Path] = None

        self.pending_locations: List[str] = []
        self.source: Gst.Element = None

        self.typefind: Gst.Element = Gst.ElementFactory.make('typefind')
        self.typefind.connect('have-type', self.on_typefind_have_type)
        self.add(self.typefind)

        self._elements: List[Gst.Element] = []

        self.capsfilter: Gst.Element = Gst.ElementFactory.make('capsfilter')
        self.add(self.capsfilter)
        self.capssetter: Gst.Element = Gst.ElementFactory.make('capssetter')
        self.add(self.capssetter)
        assert self.capsfilter.link(self.capssetter)

        self.src_pad: Gst.GhostPad = Gst.GhostPad.new(
            'src', self.capssetter.get_static_pad('src')
        )
        self.add_pad(self.src_pad)
        self.src_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.EOS: self.on_eos},
        )

    def do_set_state(self, state: Gst.State):
        self.logger.info('Changing state from %s to %s', self.current_state, state)
        if self.current_state == Gst.State.NULL and state != Gst.State.NULL:
            self.validate_properties()

            # TODO: check file type for HTTP location
            if isinstance(self.location, Path):
                self.pending_locations = self.list_files()
                self.source: Gst.Element = Gst.ElementFactory.make('filesrc')
            elif self.download_path:
                download_filepath = self.get_download_filepath(self.location)
                if download_filepath.exists():
                    self.logger.info(
                        '%r already downloaded to %r', self.location, download_filepath
                    )
                    self.pending_locations = [str(download_filepath)]
                    self.source: Gst.Element = Gst.ElementFactory.make('filesrc')
                else:
                    self.logger.info(
                        'Downloading %r to %r', self.location, download_filepath
                    )
                    download_filepath.parent.mkdir(parents=True, exist_ok=True)
                    download_file = open(download_filepath, 'wb')
                    self.pending_locations = [self.location]
                    if self.loop_file:
                        self.pending_locations.append(str(download_filepath))
                    self.source: Gst.Element = Gst.ElementFactory.make('souphttpsrc')
                    self.source.get_static_pad('src').add_probe(
                        Gst.PadProbeType.BUFFER | Gst.PadProbeType.EVENT_DOWNSTREAM,
                        self.download_file_probe,
                        download_file,
                    )
            else:
                self.pending_locations = [self.location]
                self.source: Gst.Element = Gst.ElementFactory.make('souphttpsrc')

            next_location = self.pop_next_location()
            self.logger.info('Reading file %s', next_location)
            self.source.set_property('location', next_location)
            self.source.set_state(state)
            self.add(self.source)
            assert self.source.link(self.typefind)

        return Gst.Bin.do_set_state(self, state)

    def validate_properties(self):
        assert self.location is not None, '"location" property is required'
        assert self.file_type is not None, '"file-type" property is required'
        if self.download_path is not None:
            if self.download_path.exists():
                assert (
                    self.download_path.is_dir()
                ), '"download-path" must be a directory'
        if self.loop_file:
            assert (
                self.file_type == FileType.VIDEO
            ), f'Only "file-type={FileType.VIDEO.value}" is allowed when "loop-file" is enabled'
            if not isinstance(self.location, Path):
                assert (
                    self.download_path is not None
                ), '"download-path" property is required when "loop-file" is enabled'

    def get_download_filepath(self, location: str):
        parsed_location = urlparse(location)
        return (self.download_path / parsed_location.path.lstrip('/')).absolute()

    def download_file_probe(
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        file: BinaryIO,
    ):
        if info.type & Gst.PadProbeType.BUFFER:
            buffer: Gst.Buffer = info.get_buffer()
            mapinfo: Gst.MapInfo
            mapped, mapinfo = buffer.map(Gst.MapFlags.READ)
            assert mapped, f'Failed to map buffer {buffer}'
            try:
                self.logger.debug(
                    'Writing %s bytes at offset %s to %r',
                    mapinfo.size,
                    buffer.offset,
                    file.name,
                )
                file.seek(buffer.offset)
                file.write(mapinfo.data)
            finally:
                buffer.unmap(mapinfo)

            return Gst.PadProbeReturn.OK

        if info.type & Gst.PadProbeType.EVENT_DOWNSTREAM:
            event: Gst.Event = info.get_event()
            if event.type == Gst.EventType.EOS:
                self.logger.info('Flushing %r', file.name)
                file.flush()
                file.close()
                return Gst.PadProbeReturn.REMOVE

        return Gst.PadProbeReturn.PASS

    def list_files(self):
        assert self.location.exists(), f'No such file or directory "{self.location}"'
        if self.location.is_dir():
            assert (
                not self.loop_file
            ), f'Specifying directory as location is not allowed when "loop-file" is enabled'
            all_files = sorted(
                (f for f in self.location.iterdir() if f.is_file()),
                key=(
                    (lambda x: x.stat().st_mtime_ns)
                    if self.sort_by_time
                    else (lambda x: x.name)
                ),
            )
            self.logger.info(
                'Found %s files in directory %s',
                len(all_files),
                self.location.absolute(),
            )
            # TODO: don't raise error, send EOS immediately
            assert all_files, f'Directory "{self.location}" is empty'
        elif self.location.is_file():
            all_files = [self.location]
        else:
            raise RuntimeError(f'"{self.location}" is not file or directory')

        files = [
            Path(filepath)
            for filepath, mime_type in parse_file_types(all_files)
            if MIME_TYPE_REGEX[self.file_type].fullmatch(mime_type) is not None
        ]
        self.logger.info('Reading %s %s files', len(files), self.file_type.value)
        assert files, f'No {self.file_type.value} files found'

        return [str(f.absolute()) for f in files]

    def on_typefind_have_type(
        self,
        typefind: Gst.Element,
        probability: int,
        caps: Gst.Caps,
    ):
        """Handles caps found by typefind."""

        self.logger.debug(
            'Got "have-type" signal on %s. Probability: %s Caps: %s.',
            typefind.get_name(),
            probability,
            caps,
        )

        if not self.attach_parser(typefind.get_static_pad('src'), caps):
            self.logger.debug(
                'Attaching demuxer to element %s with caps %s',
                typefind.get_name(),
                caps.to_string(),
            )
            if caps[0].get_name().endswith('mpeg'):
                self.logger.debug('Using `mpegpsdemux` demuxer')
                demuxer: Gst.Element = Gst.ElementFactory.make('mpegpsdemux')
            else:
                self.logger.debug('Using `qtdemux` demuxer')
                demuxer: Gst.Element = Gst.ElementFactory.make('qtdemux')
            demuxer.connect('pad-added', self.on_pad_added)
            self.add(demuxer)
            self._elements.append(demuxer)
            assert typefind.link(demuxer)
            demuxer.sync_state_with_parent()

    def on_pad_added(self, element: Gst.Element, pad: Gst.GhostPad):
        self.logger.debug('Added pad %s.%s', element.get_name(), pad.get_name())
        caps = pad.get_current_caps()
        if caps is None:
            pad.add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                on_pad_event,
                {Gst.EventType.CAPS: self.on_caps_change},
            )
        else:
            self.attach_parser(pad, caps)

    def on_caps_change(self, pad: Gst.Pad, event: Gst.Event):
        caps: Gst.Caps = event.parse_caps()
        self.logger.debug(
            'Caps on pad %s changed to %s', pad.get_name(), caps.to_string()
        )
        self.attach_parser(pad, caps)
        return Gst.PadProbeReturn.OK

    def attach_parser(self, pad: Gst.Pad, caps: Gst.Caps):
        try:
            self.logger.debug(f"Try to find codec for `{caps[0].get_name()}`")
            codec = CODEC_BY_CAPS_NAME[caps[0].get_name()]
        except KeyError:
            self.logger.debug(
                'Pad %s.%s has caps %s. Not attaching parser to it.',
                pad.get_parent().get_name(),
                pad.get_name(),
                caps.to_string(),
            )
            return False

        self.logger.debug(
            'Attaching parser to pad %s.%s with caps %s',
            pad.get_parent().get_name(),
            pad.get_name(),
            caps.to_string(),
        )

        parser: Gst.Element = Gst.ElementFactory.make(codec.value.parser)
        if codec.value.parser in ['h264parse', 'h265parse']:
            # Send VPS, SPS and PPS with every IDR frame
            # h26xparse cannot start stream without VPS, SPS or PPS in the first frame
            self.logger.debug('Set config-interval of %s to %s', parser.get_name(), -1)
            parser.set_property('config-interval', -1)
        self.add(parser)
        self._elements.append(parser)
        assert pad.link(parser.get_static_pad('sink')) == Gst.PadLinkReturn.OK

        filtered_caps = Gst.Caps.from_string(codec.value.caps_with_params)
        self.capsfilter.set_property('caps', filtered_caps)
        modified_caps = Gst.Caps.from_string(filtered_caps[0].get_name())
        if self.file_type == FileType.PICTURE:
            # jpegparse allows only framerate=1/1
            # pngparse doesn't set framerate
            modified_caps.set_value(
                'framerate',
                Gst.Fraction(self.framerate.numerator, self.framerate.denominator),
            )
        self.capssetter.set_property('caps', modified_caps)
        assert parser.link(self.capsfilter)
        assert self.src_pad.set_active(True)
        self.send_file_location(self.source.get_property('location'))
        self.capssetter.sync_state_with_parent()
        self.capsfilter.sync_state_with_parent()
        parser.sync_state_with_parent()

        return True

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the metadata required to specify parameters
        """
        if prop.name == 'location':
            return str(self.location)
        if prop.name == 'file-type':
            return self.file_type.value
        if prop.name == 'framerate':
            return f'{self.framerate.numerator}/{self.framerate.denominator}'
        if prop.name == 'sort-by-time':
            return self.sort_by_time
        if prop.name == 'loop-file':
            return self.loop_file
        if prop.name == 'download-path':
            return str(self.download_path)
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the metadata required to specify parameters
        :param value: new value for param, type dependents on param
        """
        if prop.name == 'location':
            if value.startswith('http://') or value.startswith('https://'):
                self.location = value
            else:
                self.location = Path(value)
        elif prop.name == 'file-type':
            assert self.current_state == Gst.State.NULL
            self.file_type = FileType(value)
        elif prop.name == 'framerate':
            try:
                self.framerate = Fraction(value)
            except (ZeroDivisionError, ValueError) as e:
                raise AttributeError(f'Invalid property {prop.name}: {e}.') from e
            self.frame_duration = int(Gst.SECOND / self.framerate)
        elif prop.name == 'sort-by-time':
            self.sort_by_time = value
        elif prop.name == 'loop-file':
            self.loop_file = value
        elif prop.name == 'download-path':
            self.download_path = Path(value)
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def on_eos(self, pad: Gst.Pad, event: Gst.Event):
        self.logger.debug(
            'Got EOS from pad %s of %s', pad.get_name(), pad.parent.get_name()
        )
        if not self.pending_locations:
            self.logger.info('No files left in %s', self.location)
            return Gst.PadProbeReturn.OK
        self.logger.debug('%s locations left', len(self.pending_locations))
        GLib.idle_add(self.start_next_file, self.pop_next_location())

        return Gst.PadProbeReturn.HANDLED

    def start_next_file(self, file_location: str):
        self.logger.info('Reading from location %s', file_location)

        self.source.set_state(Gst.State.READY)
        self.typefind.set_state(Gst.State.READY)
        assert self.src_pad.set_active(False)

        for elem in self._elements:
            self.logger.debug('Removing element %s', elem.get_name())
            elem.set_locked_state(True)
            elem.set_state(Gst.State.NULL)
            self.remove(elem)
        self._elements = []

        if self.source.get_factory().get_name() != 'filesrc':
            self.logger.info('Remove element %r', self.source.get_name())
            self.source.set_locked_state(True)
            self.source.set_state(Gst.State.NULL)
            self.remove(self.source)
            self.source: Gst.Element = Gst.ElementFactory.make('filesrc')
            self.logger.info('Add element %r', self.source.get_name())
            self.add(self.source)
            assert self.source.link(self.typefind)

        self.logger.info('Set location %s', file_location)
        self.source.set_property('location', file_location)
        self.typefind.sync_state_with_parent()
        self.source.sync_state_with_parent()

        return False

    def pop_next_location(self):
        if self.loop_file and len(self.pending_locations) == 1:
            return self.pending_locations[0]
        else:
            return self.pending_locations.pop(0)

    def send_file_location(self, file_location: str):
        tag_list: Gst.TagList = Gst.TagList.new_empty()
        tag_list.add_value(Gst.TagMergeMode.APPEND, Gst.TAG_LOCATION, file_location)
        tag_event: Gst.Event = Gst.Event.new_tag(tag_list)
        self.src_pad.push_event(tag_event)

    def set_frame_duration(self, pad: Gst.Pad, info: Gst.PadProbeInfo):
        buffer: Gst.Buffer = info.get_buffer()
        buffer.duration = self.frame_duration
        return Gst.PadProbeReturn.OK


def parse_file_types(files: List[Path]):
    output = subprocess.check_output(
        ['file', '--no-pad', '--mime-type'] + [str(x) for x in files]
    )
    return [x.rsplit(': ', 1) for x in output.decode().strip().split('\n')]


# register plugin
GObject.type_register(MediaFilesSrcBin)
__gstelementfactory__ = (
    MediaFilesSrcBin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    MediaFilesSrcBin,
)
