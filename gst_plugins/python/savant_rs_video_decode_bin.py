"""SavantRsVideoDecodeBin element."""

import time
from dataclasses import dataclass
from threading import Event, Lock
from typing import Dict, Optional

from pygstsavantframemeta import add_pad_probe_to_move_frame
from savant_rs.pipeline2 import VideoPipeline

from gst_plugins.python.savant_rs_video_demux import SAVANT_RS_VIDEO_DEMUX_PROPERTIES
from savant.gstreamer import GLib, GObject, Gst  # noqa:F401
from savant.gstreamer.codecs import CODEC_BY_CAPS_NAME, Codec
from savant.gstreamer.utils import on_pad_event, pad_to_source_id
from savant.utils.logging import LoggerMixin
from savant.utils.platform import is_aarch64

OUT_CAPS = Gst.Caps.from_string('video/x-raw(memory:NVMM);video/x-raw')
DEFAULT_PASS_EOS = True
# Default values of "queue" element
DEFAULT_DECODER_QUEUE_LENGTH = 200
DEFAULT_DECODER_QUEUE_SIZE = 10485760

NESTED_DEMUX_PROPERTIES = {
    k: v
    for k, v in SAVANT_RS_VIDEO_DEMUX_PROPERTIES.items()
    if k
    in [
        'source-timeout',
        'source-eviction-interval',
        'max-parallel-streams',
        'zeromq-reader',
    ]
}
SAVANT_RS_VIDEO_DECODE_BIN_PROPERTIES = {
    'low-latency-decoding': (
        bool,
        'Enable low-latency mode for decoding',
        'Enable low-latency mode for decoding.'
        'I.e. disable decoded picture buffer (DPB).',
        True,
        GObject.ParamFlags.READWRITE,
    ),
    'pass-eos': (
        bool,
        'Whether to pass EOS event downstream or not',
        'Whether to pass EOS event downstream or not',
        DEFAULT_PASS_EOS,
        GObject.ParamFlags.READWRITE,
    ),
    'pipeline': (
        object,
        'VideoPipeline object from savant-rs.',
        'VideoPipeline object from savant-rs.',
        GObject.ParamFlags.READWRITE,
    ),
    'pipeline-demux-stage-name': (
        str,
        'Name of the pipeline stage for demuxer.',
        'Name of the pipeline stage for demuxer.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'pipeline-decoder-stage-name': (
        str,
        'Name of the pipeline stage for decoder.',
        'Name of the pipeline stage for decoder.',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'decoder-queue-length': (
        int,
        'Length of the queue before decoder in frames.',
        'Length of the queue before decoder in frames (0 - no limit).',
        0,
        GObject.G_MAXINT,
        DEFAULT_DECODER_QUEUE_LENGTH,
        GObject.ParamFlags.READWRITE,
    ),
    'decoder-queue-byte-size': (
        int,
        'Size of the queue before decoder in bytes.',
        'Size of the queue before decoder in bytes (0 - no limit).',
        0,
        GObject.G_MAXINT,
        DEFAULT_DECODER_QUEUE_SIZE,
        GObject.ParamFlags.READWRITE,
    ),
    **NESTED_DEMUX_PROPERTIES,
}
SAVANT_RS_VIDEO_DECODE_BIN_SINK_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink',
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.new_any(),
)
SAVANT_RS_VIDEO_DECODE_BIN_SRC_PAD_TEMPLATE = Gst.PadTemplate.new(
    'src_%s',
    Gst.PadDirection.SRC,
    Gst.PadPresence.SOMETIMES,
    OUT_CAPS,
)


@dataclass
class BranchInfo:
    """Info about source."""

    source_id: str
    lock: Event
    caps: Optional[Gst.Caps] = None
    codec: Optional[Codec] = None
    decoder: Optional[Gst.Element] = None
    src_pad: Optional[Gst.GhostPad] = None
    pad_added_to_bin: bool = False

    @property
    def caps_name(self):
        """Gst.Caps name."""
        return self.caps[0].get_name()


class SavantRsVideoDecodeBin(LoggerMixin, Gst.Bin):
    """'Decodes savant-rs video stream."""

    GST_PLUGIN_NAME = 'savant_rs_video_decode_bin'

    __gstmetadata__ = (
        'Savant-rs video decode bin',
        'Bin/Decoder',
        'Decodes savant-rs video stream. '
        'Outputs decoded video frames to src pad "src_<source_id>".',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        SAVANT_RS_VIDEO_DECODE_BIN_SINK_PAD_TEMPLATE,
        SAVANT_RS_VIDEO_DECODE_BIN_SRC_PAD_TEMPLATE,
    )

    __gproperties__ = SAVANT_RS_VIDEO_DECODE_BIN_PROPERTIES

    __gsignals__ = {'shutdown': (GObject.SignalFlags.RUN_LAST, None, ())}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._elem_to_branch: Dict[Gst.Element, BranchInfo] = {}
        self._branches_lock = Lock()
        self._branches: Dict[str, BranchInfo] = {}

        # properties
        self._low_latency_decoding = False
        self._pass_eos = DEFAULT_PASS_EOS
        self._video_pipeline: Optional[VideoPipeline] = None
        self._pipeline_decoder_stage_name: Optional[str] = None
        self._decoder_queue_length = DEFAULT_DECODER_QUEUE_LENGTH
        self._decoder_queue_byte_size = DEFAULT_DECODER_QUEUE_SIZE

        self._demuxer: Gst.Element = Gst.ElementFactory.make('savant_rs_video_demux')
        self._demuxer.set_property('eos-on-timestamps-reset', True)
        self.add(self._demuxer)
        self._demuxer.connect('pad-added', self.on_pad_added)
        self._demuxer.connect('shutdown', self.on_shutdown)
        self._max_parallel_streams: int = self._demuxer.get_property(
            'max-parallel-streams'
        )

        self._sink_pad: Gst.GhostPad = Gst.GhostPad.new(
            'sink', self._demuxer.get_static_pad('sink')
        )
        self.add_pad(self._sink_pad)
        self._sink_pad.set_active(True)

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the metadata
            required to specify parameters
        """
        if prop.name == 'low-latency-decoding':
            return self._low_latency_decoding
        if prop.name == 'pass-eos':
            return self._pass_eos
        if prop.name == 'max-parallel-streams':
            return self._max_parallel_streams
        if prop.name == 'pipeline':
            return self._video_pipeline
        if prop.name == 'pipeline-demux-stage-name':
            return self._demuxer.get_property('pipeline-stage-name')
        if prop.name == 'pipeline-decoder-stage-name':
            return self._pipeline_decoder_stage_name
        if prop.name == 'decoder-queue-length':
            return self._decoder_queue_length
        if prop.name == 'decoder-queue-byte-size':
            return self._decoder_queue_byte_size
        if prop.name in NESTED_DEMUX_PROPERTIES:
            return self._demuxer.get_property(prop.name)
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the metadata
            required to specify parameters
        :param value: new value for param, type dependents on param
        """
        if prop.name == 'low-latency-decoding':
            self._low_latency_decoding = value
        elif prop.name == 'pass-eos':
            self._pass_eos = value
        elif prop.name == 'max-parallel-streams':
            self._max_parallel_streams = value
            self._demuxer.set_property(prop.name, value)
        elif prop.name == 'pipeline':
            self._video_pipeline = value
            self._demuxer.set_property(prop.name, value)
        elif prop.name == 'pipeline-demux-stage-name':
            self._demuxer.set_property('pipeline-stage-name', value)
        elif prop.name == 'pipeline-decoder-stage-name':
            self._pipeline_decoder_stage_name = value
        elif prop.name == 'decoder-queue-length':
            self._decoder_queue_length = value
        elif prop.name == 'decoder-queue-byte-size':
            self._decoder_queue_byte_size = value
        elif prop.name in NESTED_DEMUX_PROPERTIES:
            self._demuxer.set_property(prop.name, value)
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def do_handle_message(self, message: Gst.Message):
        """Processes posted messages.

        Removes source if it posted a change to NULL state.
        """
        self.logger.debug(
            'Received message %s from %s',
            message.type,
            message.src.get_name(),
        )

        if message.type == Gst.MessageType.STRUCTURE_CHANGE:
            self.logger.debug(
                'Received STRUCTURE_CHANGE message from %s: %s',
                message.src.get_name(),
                message.parse_structure_change(),
            )
            # Cannot pass STRUCTURE_CHANGE to Gst.Bin.do_handle_message()
            # gst_structure_set_parent_refcount: assertion 'refcount != NULL' failed
            return

        if (
            message.type == Gst.MessageType.STATE_CHANGED
            and message.src in self._elem_to_branch
        ):
            return self.on_decoder_state_change(message)

        if message.type == Gst.MessageType.ERROR:
            return self.on_error_message(message)

        return Gst.Bin.do_handle_message(self, message)

    def on_decoder_state_change(self, message):
        old, new, pending = message.parse_state_changed()
        self.logger.debug(
            'State of element %s changed from %s to %s (%s pending)',
            message.src.get_name(),
            old,
            new,
            pending,
        )
        if new == Gst.State.NULL:
            self.logger.debug('Removing element %s', message.src.get_name())
            self.remove(message.src)

        return Gst.Bin.do_handle_message(self, message)

    def on_error_message(self, message: Gst.Message):
        err, debug = message.parse_error()
        self.logger.warning(
            'Received error from %s: %s. Debug info: %s.',
            message.src.get_name(),
            err,
            debug,
        )
        src: Gst.Element = message.src
        for pad in src.iterate_pads():
            caps: Gst.Caps = pad.get_current_caps()
            self.logger.warning(
                'Pad %s.%s has caps %s', src.get_name(), pad.get_name(), caps
            )
        branch = self._elem_to_branch.get(src)
        while src is not None and branch is None:
            src = src.get_parent()
            if src is not None:
                branch = self._elem_to_branch.get(src)

        if branch is not None:
            GLib.idle_add(self._remove_branch, branch)
            # Drop the message to prevent it from being handled by the parent element
            return

        return Gst.Bin.do_handle_message(self, message)

    def on_pad_added(self, element: Gst.Element, new_pad: Gst.Pad):
        """Handle adding new source."""

        self.logger.debug(
            'Added pad %s on element %s', new_pad.get_name(), element.get_name()
        )
        if self._video_pipeline:
            add_pad_probe_to_move_frame(
                new_pad,
                self._video_pipeline,
                self._pipeline_decoder_stage_name,
            )
        source_id = pad_to_source_id(new_pad)
        caps = new_pad.get_pad_template_caps()
        new_pad.add_probe(
            Gst.PadProbeType.BLOCK_DOWNSTREAM,
            self._add_branch,
            source_id,
            caps,
        )

    def _add_branch(
        self,
        pad: Gst.Pad,
        probe_info: Gst.PadProbeInfo,
        source_id: str,
        caps: Gst.Caps,
    ):
        self.logger.info('Adding branch with source %s', source_id)
        pad.remove_probe(probe_info.id)

        branch = self._branches.get(source_id)
        if branch is not None:
            while not branch.lock.wait(5):
                self.logger.debug(
                    'Waiting resources for source %s to be released.', source_id
                )
            branch.lock.clear()
        else:
            branch = BranchInfo(source_id=source_id, lock=Event())
        while True:
            with self._branches_lock:
                if (
                    self._max_parallel_streams
                    and len(self._branches) >= self._max_parallel_streams
                ):
                    # savant_rs_video_demux already sent EOS for some stream
                    # and adding a new one, but the former stream did not complete
                    # in savant_rs_video_decode_bin yet.
                    self.logger.warning(
                        'Reached maximum number of streams: %s. '
                        'Waiting resources for source %s.',
                        self._max_parallel_streams,
                        source_id,
                    )
                else:
                    self._branches[source_id] = branch
                    break
            time.sleep(5)

        branch.caps = caps
        branch.codec = CODEC_BY_CAPS_NAME[caps[0].get_name()]
        branch.src_pad = Gst.GhostPad.new_no_target(
            pad.get_name(), Gst.PadDirection.SRC
        )

        branch.decoder = self.build_decoder(branch)
        self._elem_to_branch[branch.decoder] = branch

        branch.src_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.EOS: self.on_src_pad_eos},
            branch,
        )
        self.add(branch.decoder)
        branch.decoder.sync_state_with_parent()
        assert pad.link(branch.decoder.get_static_pad('sink')) == Gst.PadLinkReturn.OK

        self.set_state(Gst.State.PLAYING)
        self.logger.info('Branch with source %s added', source_id)

        return Gst.PadProbeReturn.OK

    def on_decodebin_pad_added(
        self,
        decodebin: Gst.Element,
        new_pad: Gst.Pad,
        branch: BranchInfo,
    ):
        """Target a branch src ghost pad pad to an added decoder pad."""
        self.logger.debug(
            'Attaching pad %s.%s to ghost pad %s',
            decodebin.get_name(),
            new_pad.get_name(),
            branch.src_pad.get_name(),
        )

        decoder: Gst.Bin = decodebin.get_parent()
        decoder_pad: Gst.GhostPad = Gst.GhostPad.new(new_pad.get_name(), new_pad)
        decoder.add_pad(decoder_pad)
        decoder_pad.set_active(True)

        self.add_pad(branch.src_pad)
        branch.pad_added_to_bin = True
        assert branch.src_pad.set_target(decoder_pad)
        assert branch.src_pad.set_active(True)
        self.logger.debug(
            'Pad %s.%s attached to ghost pad %s via %s.%s',
            decodebin.get_name(),
            new_pad.get_name(),
            branch.src_pad.get_name(),
            decoder.get_name(),
            decoder_pad.get_name(),
        )

    def on_src_pad_eos(self, pad: Gst.Pad, event: Gst.Event, branch: BranchInfo):
        """Attaches a callback to remove a branch."""
        self.logger.debug(
            'Got EOS from pad %s of %s',
            pad.get_name(),
            pad.parent.get_name() if pad.parent is not None else None,
        )
        if self._pass_eos:
            peer: Gst.Pad = pad.get_peer()
            peer.send_event(Gst.Event.new_eos())
        GLib.idle_add(self._remove_branch, branch)
        return Gst.PadProbeReturn.DROP

    def _remove_branch(self, branch: BranchInfo):
        """Remove a branch."""
        self.logger.info('Removing branch with source %s', branch.source_id)
        self.logger.debug('Removing pad %s', branch.src_pad.get_name())
        branch.decoder.set_locked_state(True)
        # do_handle_message deletes branch.decoder when its state changed to NULL
        self.logger.debug('Setting element %s to state NULL', branch.decoder.get_name())
        branch.decoder.set_state(Gst.State.NULL)
        self.logger.debug('Set element %s to state NULL', branch.decoder.get_name())

        self.logger.debug('Setting state of the bin to PLAYING')
        self.set_state(Gst.State.PLAYING)
        self.logger.info('Branch with source %s removed', branch.source_id)

        return False

    def do_element_removed(self, elem: Gst.Element):
        """Release a removed element resources."""
        self.logger.debug('Element %s has been removed', elem.get_name())
        branch = self._elem_to_branch.pop(elem, None)
        if branch is None:
            return
        self.logger.debug('Resources of source %s has been released', branch.source_id)
        if branch.pad_added_to_bin:
            self.remove_pad(branch.src_pad)
        del self._branches[branch.source_id]
        branch.lock.set()
        self.logger.debug(
            'Lock %s of source %s has been released', branch.lock, branch.source_id
        )

    def build_decoder(self, branch: BranchInfo):
        self.logger.debug('Building decoder for source %s', branch.source_id)
        queue_name = f'in-queue-{branch.source_id}'
        decodebin_name = f'decodebin-{branch.source_id}'
        decoder: Gst.Bin = Gst.parse_bin_from_description(
            f'queue name={queue_name} ! decodebin name={decodebin_name}', False
        )

        in_queue: Gst.Element = decoder.get_by_name(queue_name)
        in_queue.set_property('max-size-buffers', self._decoder_queue_length)
        in_queue.set_property('max-size-bytes', self._decoder_queue_byte_size)
        in_queue.set_property('max-size-time', 0)

        decodebin: Gst.Element = decoder.get_by_name(decodebin_name)
        decodebin.set_property('max-size-buffers', self._decoder_queue_length)
        decodebin.set_property('max-size-bytes', self._decoder_queue_byte_size)
        decodebin.set_property('max-size-time', 0)

        # https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_FAQ.html#on-jetson-platform-i-get-same-output-when-multiple-jpeg-images-are-fed-to-nvv4l2decoder-using-multifilesrc-plugin-why-is-that
        if branch.codec == Codec.JPEG and is_aarch64():

            def on_add_element(
                bin: Gst.Bin,
                elem: Gst.Element,
            ):
                if elem.get_factory().get_name() == 'nvv4l2decoder':
                    self.logger.debug('Added mjpeg=true for nvv4l2decoder element')
                    elem.set_property('mjpeg', 1)

            decodebin.connect('element-added', on_add_element)

        self.logger.debug('Configuring decodebin for source %s', branch.source_id)
        # TODO: configure low-latency decoding
        if branch.codec == Codec.H264:
            # pipeline hangs up when caps specified with all the properties
            decodebin.set_property(
                'sink-caps', Gst.Caps.from_string(branch.codec.value.caps_with_params)
            )
        else:
            decodebin.set_property('sink-caps', branch.caps)
        decodebin.set_property('caps', OUT_CAPS)
        self.logger.debug('Built decoder for source %s.', branch.source_id)

        decodebin.connect('pad-added', self.on_decodebin_pad_added, branch)

        sink_pad: Gst.GhostPad = Gst.GhostPad.new(
            'sink', in_queue.get_static_pad('sink')
        )
        decoder.add_pad(sink_pad)
        sink_pad.set_active(True)

        return decoder

    def on_shutdown(self, element: Gst.Element):
        """Handle shutdown signal."""

        self.logger.debug(
            'Received shutdown signal from %s. Passing it downstream.',
            element.get_name(),
        )
        self.emit('shutdown')


# register plugin
GObject.type_register(SavantRsVideoDecodeBin)
__gstelementfactory__ = (
    SavantRsVideoDecodeBin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SavantRsVideoDecodeBin,
)
