from dataclasses import dataclass
from threading import Event
from typing import Dict, Optional

from pygstsavantframemeta import gst_buffer_get_savant_frame_meta
from savant_rs.pipeline2 import (
    StageFunction,
    VideoPipeline,
    VideoPipelineConfiguration,
    VideoPipelineStagePayloadType,
)

from gst_plugins.python.zeromq_src import ZEROMQ_SRC_PROPERTIES
from savant.gstreamer import GLib, GObject, Gst
from savant.gstreamer.utils import on_pad_event, pad_to_source_id
from savant.utils.logging import LoggerMixin
from savant.utils.platform import is_aarch64

NESTED_ZEROMQ_SRC_PROPERTIES = {
    k: v
    for k, v in ZEROMQ_SRC_PROPERTIES.items()
    if k not in ['pipeline', 'pipeline-stage-name']
}


@dataclass
class BranchInfo:
    """Info about source."""

    source_id: str
    sink: Optional[Gst.Bin] = None


class SavantRsVideoPlayer(LoggerMixin, Gst.Bin):
    """Plays savant-rs video on display."""

    GST_PLUGIN_NAME = 'savant_rs_video_player'

    __gstmetadata__ = (
        'Savant-rs video player',
        'Bin/Player',
        'Deserializes savant-rs video and plays it on display',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = ()

    __gproperties__ = {
        **NESTED_ZEROMQ_SRC_PROPERTIES,
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

        source_stage_name = 'video-player-source'
        demux_stage_name = 'video-player-demux'
        decoder_stage_name = 'video-player-decoder'
        self._video_pipeline: VideoPipeline = VideoPipeline(
            'video-player',
            [
                (
                    source_stage_name,
                    VideoPipelineStagePayloadType.Frame,
                    StageFunction.none(),
                    StageFunction.none(),
                ),
                (
                    demux_stage_name,
                    VideoPipelineStagePayloadType.Frame,
                    StageFunction.none(),
                    StageFunction.none(),
                ),
                (
                    decoder_stage_name,
                    VideoPipelineStagePayloadType.Frame,
                    StageFunction.none(),
                    StageFunction.none(),
                ),
            ],
            VideoPipelineConfiguration(),
        )

        self._source: Gst.Element = Gst.ElementFactory.make('zeromq_src')
        self._source.set_property('pipeline', self._video_pipeline)
        self._source.set_property('pipeline-stage-name', source_stage_name)
        self.add(self._source)

        self._queue: Gst.Element = Gst.ElementFactory.make('queue')
        self._queue.set_property('max-size-time', 0)
        self.add(self._queue)

        self._decoder: Gst.Element = Gst.ElementFactory.make(
            'savant_rs_video_decode_bin'
        )
        self._decoder.set_property('pass-eos', True)
        self._decoder.set_property('pipeline', self._video_pipeline)
        self._decoder.set_property('pipeline-demux-stage-name', decoder_stage_name)
        self._decoder.set_property('pipeline-decoder-stage-name', decoder_stage_name)
        self._decoder.set_property('source-timeout', 10)
        self._decoder.set_property('source-eviction-interval', 1)
        self.add(self._decoder)
        self._decoder.connect('pad-added', self.on_pad_added)

        assert self._source.link(self._queue), f'Failed to link source to queue'
        assert self._queue.link(self._decoder), f'Failed to link queue to decoder'

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the metadata required to specify parameters
        """
        if prop.name == 'sync':
            return self._sync
        if prop.name == 'closing-delay':
            return self._closing_delay
        if prop.name in NESTED_ZEROMQ_SRC_PROPERTIES:
            return self._source.get_property(prop.name)
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
        elif prop.name in NESTED_ZEROMQ_SRC_PROPERTIES:
            self._source.set_property(prop.name, value)
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
        caps: Gst.Caps = new_pad.get_current_caps()
        if caps is not None:
            self.logger.info('Waiting for caps on pad %s', new_pad.get_name())
            new_pad.add_probe(
                Gst.PadProbeType.BLOCK_DOWNSTREAM, self._add_branch, branch
            )
        else:
            self.logger.info('Caps on pad %s: %s', new_pad.get_name(), caps)
            new_pad.add_probe(
                Gst.PadProbeType.EVENT_DOWNSTREAM,
                on_pad_event,
                {Gst.EventType.CAPS: self.on_caps_change},
                branch,
            )
        new_pad.add_probe(Gst.PadProbeType.BUFFER, self.delete_frame_from_pipeline)

    def delete_frame_from_pipeline(self, pad: Gst.Pad, info: Gst.PadProbeInfo):
        buffer: Gst.Buffer = info.get_buffer()
        savant_frame_meta = gst_buffer_get_savant_frame_meta(buffer)
        source_id = pad_to_source_id(pad)
        if savant_frame_meta is None:
            self.logger.warning(
                'Source %s. No Savant Frame Metadata found on buffer with PTS %s.',
                source_id,
                buffer.pts,
            )
            return Gst.PadProbeReturn.PASS

        self.logger.debug(
            'Deleting frame %s/%s and PTS %s from pipeline',
            source_id,
            savant_frame_meta.idx,
            buffer.pts,
        )
        self._video_pipeline.delete(savant_frame_meta.idx)

        return Gst.PadProbeReturn.OK

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

        self.logger.debug('Creating sink bin for source %s', branch.source_id)
        videosink_name = f'videosink_{branch.source_id}'
        sink_bin_elements = [
            'queue',
            'nvvideoconvert',
            'capsfilter caps=video/x-raw(memory:NVMM)',
            'adjust_timestamps',
        ]
        if is_aarch64():
            sink_bin_elements.append('nvegltransform')
        sink_bin_elements.append(f'nveglglessink name={videosink_name}')
        branch.sink = Gst.parse_bin_from_description(
            ' ! '.join(sink_bin_elements),
            True,
        )
        self.logger.debug(
            'Sink bin for source %s created: %s',
            branch.source_id,
            branch.sink.get_name(),
        )
        self.add(branch.sink)
        self.logger.debug('Sink bin for source %s added', branch.source_id)
        sink: Gst.Element = branch.sink.get_by_name(videosink_name)
        sink.set_property('sync', self._sync)
        if self._sync:
            sink.set_property(
                'ts-offset', self.get_clock().get_time() - self.get_base_time()
            )
        branch.sink.sync_state_with_parent()
        assert pad.link(branch.sink.get_static_pad('sink')) == Gst.PadLinkReturn.OK
        self.logger.debug(
            'Pad %s linked to sink %s',
            pad.get_name(),
            branch.sink.get_name(),
        )

        self._elem_to_source_id[branch.sink] = branch.source_id
        self._locks[branch.source_id] = Event()
        sink.get_static_pad('sink').add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.EOS: self.on_sink_pad_eos},
            branch,
        )
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
        branch.sink.set_locked_state(True)
        branch.sink.set_state(Gst.State.NULL)
        self.logger.info('Removing element %s', branch.sink.get_name())
        self.remove(branch.sink)

        self.logger.info('Branch with source %s removed', branch.source_id)

        return False

    def on_caps_change(self, pad: Gst.Pad, event: Gst.Event, branch: BranchInfo):
        caps: Gst.Caps = event.parse_caps()
        self.logger.info(
            'Caps on pad %s changed to %s', pad.get_name(), caps.to_string()
        )

        if branch.sink is None:
            pad.add_probe(Gst.PadProbeType.BLOCK_DOWNSTREAM, self._add_branch, branch)

        return Gst.PadProbeReturn.OK


# register plugin
GObject.type_register(SavantRsVideoPlayer)
__gstelementfactory__ = (
    SavantRsVideoPlayer.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SavantRsVideoPlayer,
)
