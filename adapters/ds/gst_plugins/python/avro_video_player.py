from threading import Event
from typing import Dict, Optional

from dataclasses import dataclass

from savant.gstreamer import GLib, GObject, Gst
from savant.gstreamer.utils import LoggerMixin, on_pad_event, pad_to_source_id


@dataclass
class BranchInfo:
    """Info about source."""

    source_id: str
    sink: Optional[Gst.Bin] = None


class AvroVideoPlayer(LoggerMixin, Gst.Bin):
    """Plays avro video on display."""

    GST_PLUGIN_NAME = 'avro_video_player'

    __gstmetadata__ = (
        'Avro video player',
        'Bin/Sink/Player',
        'Deserializes avro video and plays it on display',
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
        'sync': (
            bool,
            'Sync on the clock',
            'Sync on the clock',
            False,
            GObject.ParamFlags.READWRITE,
        ),
        'closing-delay': (
            int,
            'Window closing delay',
            'Delay in seconds before closing window after the last frame',
            0,
            2147483647,
            0,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._elem_to_source_id: Dict[Gst.Element, str] = {}
        self._locks: Dict[str, Event] = {}

        # properties
        self._sync = False
        self._closing_delay = 0

        self._decoder: Gst.Element = Gst.ElementFactory.make('avro_video_decode_bin')
        self._decoder.set_property('pass-eos', True)
        self.add(self._decoder)
        self._decoder.connect('pad-added', self.on_pad_added)

        self._sink_pad: Gst.GhostPad = Gst.GhostPad.new(
            'sink', self._decoder.get_static_pad('sink')
        )
        self.add_pad(self._sink_pad)
        self._sink_pad.set_active(True)

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the metadata required to specify parameters
        """
        if prop.name == 'sync':
            return self._sync
        if prop.name == 'closing-delay':
            return self._closing_delay
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the metadata required to specify parameters
        :param value: new value for param, type dependents on param
        """
        if prop.name == 'sync':
            self._sync = value
        elif prop.name == 'closing-delay':
            self._closing_delay = value
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def do_handle_message(self, message: Gst.Message):
        if message.type == Gst.MessageType.STRUCTURE_CHANGE:
            # Cannot pass STRUCTURE_CHANGE to Gst.Bin.do_handle_message()
            # gst_structure_set_parent_refcount: assertion 'refcount != NULL' failed
            return
        if (
            message.type == Gst.MessageType.STATE_CHANGED
            and message.src in self._elem_to_source_id
        ):
            source_id = self._elem_to_source_id[message.src]
            old, new, pending = message.parse_state_changed()
            self.logger.debug(
                'State of element %s changed from %s to %s (%s pending)',
                message.src.get_name(),
                old,
                new,
                pending,
            )
            if new == Gst.State.NULL:
                self.logger.debug('Resources of source %s has been released', source_id)
                del self._elem_to_source_id[message.src]
                self._locks.pop(source_id).set()

        return Gst.Bin.do_handle_message(self, message)

    def on_pad_added(self, element: Gst.Element, new_pad: Gst.Pad):
        """Handle adding new source."""

        self.logger.info(
            'Added pad %s on element %s', new_pad.get_name(), element.get_name()
        )
        branch = BranchInfo(source_id=pad_to_source_id(new_pad))
        new_pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, self._add_branch, branch)

    def _add_branch(
        self,
        pad: Gst.Pad,
        probe_info: Gst.PadProbeInfo,
        branch: BranchInfo,
    ):
        self.logger.info('Adding branch with source %s', branch.source_id)
        pad.remove_probe(probe_info.id)

        lock = self._locks.get(branch.source_id)
        if lock is not None:
            self.logger.debug(
                'Waiting for resources of source %s to be released', branch.source_id
            )
            lock.wait()

        videosink_name = f'videosink_{branch.source_id}'
        branch.sink = Gst.parse_bin_from_description(
            ' ! '.join(
                [
                    'queue',
                    'nvvideoconvert',
                    # nveglglessink cannot negotiate with "video/x-raw(memory:NVMM)" on dGPU
                    'capsfilter caps=video/x-raw',
                    'adjust_timestamps',
                    f'autovideosink name={videosink_name}',
                ]
            ),
            True,
        )
        self.add(branch.sink)
        sink: Gst.Element = branch.sink.get_by_name(videosink_name)
        sink.set_property('sync', self._sync)
        if self._sync:
            sink.set_property(
                'ts-offset', self.get_clock().get_time() - self.get_base_time()
            )
        branch.sink.sync_state_with_parent()
        assert pad.link(branch.sink.get_static_pad('sink')) == Gst.PadLinkReturn.OK

        self._elem_to_source_id[branch.sink] = branch.source_id
        self._locks[branch.source_id] = Event()
        sink.get_static_pad('sink').add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.EOS: self.on_sink_pad_eos},
            branch,
        )
        pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.CAPS: self.on_caps_change},
            branch,
        )
        self.set_state(Gst.State.PLAYING)
        self.logger.info('Branch with source %s added', branch.source_id)

        return Gst.PadProbeReturn.OK

    def on_sink_pad_eos(self, pad: Gst.Pad, event: Gst.Event, branch: BranchInfo):
        self.logger.info(
            'Got EOS from pad %s of %s', pad.get_name(), pad.parent.get_name()
        )
        GLib.timeout_add_seconds(self._closing_delay, self._remove_branch, branch)
        return Gst.PadProbeReturn.HANDLED

    def _remove_branch(self, branch: BranchInfo):
        self.logger.info('Removing branch with source %s', branch.source_id)
        branch.sink.set_state(Gst.State.NULL)
        self.logger.info('Removing element %s', branch.sink.get_name())
        self.remove(branch.sink)

        self.set_state(Gst.State.PLAYING)
        self.logger.info('Branch with source %s removed', branch.source_id)

        return False

    def on_caps_change(self, pad: Gst.Pad, event: Gst.Event, branch: BranchInfo):
        caps: Gst.Caps = event.parse_caps()
        self.logger.info(
            'Caps on pad %s changed to %s', pad.get_name(), caps.to_string()
        )

        self.logger.debug('Set state of %s to READY', branch.sink.get_name())
        branch.sink.set_state(Gst.State.READY)
        self.logger.debug('Sync state of %s with parent', branch.sink.get_name())
        branch.sink.sync_state_with_parent()

        return Gst.PadProbeReturn.OK


# register plugin
GObject.type_register(AvroVideoPlayer)
__gstelementfactory__ = (
    AvroVideoPlayer.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    AvroVideoPlayer,
)
