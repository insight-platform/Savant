import inspect
from itertools import count
from typing import List, Optional

import pyds
from pygstsavantframemeta import (
    gst_buffer_add_savant_frame_meta,
    nvds_frame_meta_get_nvds_savant_frame_meta,
)

from gst_plugins.python.frame_tag_filter_common import build_stream_part_event
from savant.deepstream.utils import nvds_frame_meta_iterator
from savant.gstreamer import GObject, Gst
from savant.gstreamer.metadata import get_source_frame_meta
from savant.gstreamer.utils import (
    RequiredPropertyError,
    on_pad_event,
    propagate_gst_setting_error,
    required_property,
)
from savant.utils.logging import LoggerMixin

SINK_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink',
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.new_any(),
)
SRC_TAGGED_PAD_TEMPLATE = Gst.PadTemplate.new(
    'src_tagged',
    Gst.PadDirection.SRC,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.new_any(),
)
SRC_NOT_TAGGED_PAD_TEMPLATE = Gst.PadTemplate.new(
    'src_not_tagged',
    Gst.PadDirection.SRC,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.new_any(),
)


class FrameTagFilter(LoggerMixin, Gst.Element):
    """Frame tag filter.

    When frame has specified tag, it is passed to src_tagged pad,
    otherwise to src_not_tagged pad.
    """

    GST_PLUGIN_NAME = 'frame_tag_filter'

    __gstmetadata__ = (
        'Frame tag filter',
        'Demuxer',
        'Filters frames by tag. When frame has specified tag, it is passed '
        'to src_tagged pad, otherwise to src_not_tagged pad.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        SINK_PAD_TEMPLATE,
        SRC_TAGGED_PAD_TEMPLATE,
        SRC_NOT_TAGGED_PAD_TEMPLATE,
    )

    __gproperties__ = {
        'source-id': (
            str,
            'Source ID.',
            'Source ID.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'tag': (
            str,
            'Frame tag for filtering frames to be encoded.',
            'Frame tag for filtering frames to be encoded.',
            None,
            GObject.ParamFlags.READWRITE | Gst.PARAM_MUTABLE_READY,
        ),
    }

    def __init__(self):
        super().__init__()

        self.part_id_gen = count()
        self.source_id: Optional[str] = None
        self.tag: Optional[str] = None

        self.sink_pad: Gst.Pad = Gst.Pad.new_from_template(SINK_PAD_TEMPLATE, 'sink')
        self.src_pad_tagged: Gst.Pad = Gst.Pad.new_from_template(
            SRC_TAGGED_PAD_TEMPLATE, 'src_tagged'
        )
        self.src_pad_not_tagged: Gst.Pad = Gst.Pad.new_from_template(
            SRC_NOT_TAGGED_PAD_TEMPLATE, 'src_not_tagged'
        )
        self.last_tagged: Optional[bool] = None

        self.add_pad(self.sink_pad)
        self.add_pad(self.src_pad_tagged)
        self.add_pad(self.src_pad_not_tagged)

        self.sink_pad.set_chain_function_full(self.handle_buffer)
        self.sink_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.CAPS: self.on_caps},
        )

    def do_state_changed(self, old: Gst.State, new: Gst.State, pending: Gst.State):
        """Validate properties when the element is started."""

        init_state = [Gst.State.NULL, Gst.State.READY]
        if old in init_state and new not in init_state:
            try:
                required_property(self.source_id, 'source-id')
                required_property(self.tag, 'tag')
            except RequiredPropertyError as exc:
                self.logger.exception('Failed to start element: %s', exc, exc_info=True)
                frame = inspect.currentframe()
                propagate_gst_setting_error(self, frame, __file__, text=exc.args[0])

    def do_get_property(self, prop):
        """Get property callback."""

        if prop.name == 'tag':
            return self.tag
        if prop.name == 'source-id':
            return self.source_id
        raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        """Set property callback."""

        self.logger.debug('Setting property "%s" to "%s".', prop.name, value)
        if prop.name == 'tag':
            self.tag = value
        elif prop.name == 'source-id':
            self.source_id = value
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def handle_buffer(
        self,
        sink_pad: Gst.Pad,
        element: Gst.Element,
        buffer: Gst.Buffer,
    ) -> Gst.FlowReturn:
        """Route buffer either to src_tagged or src_not_tagged pad."""

        self.logger.debug('Handling buffer PTS=%s', buffer.pts)

        not_tagged_buffers = self.parse_buffer(buffer)
        tagged = not_tagged_buffers is None
        if tagged != self.last_tagged:
            self.logger.debug('Switching to tagged=%s', tagged)
            event = build_stream_part_event(next(self.part_id_gen), tagged)
            self.src_pad_not_tagged.push_event(event)
            self.src_pad_tagged.push_event(event)

        self.last_tagged = tagged
        if tagged:
            return self.push_buffer(self.src_pad_tagged, buffer)

        for not_tagged_buffer in not_tagged_buffers:
            ret = self.push_buffer(self.src_pad_not_tagged, not_tagged_buffer)
            if ret != Gst.FlowReturn.OK:
                return ret

        return Gst.FlowReturn.OK

    def parse_buffer(self, buffer: Gst.Buffer) -> Optional[List[Gst.Buffer]]:
        """Parse buffer and return list of not tagged buffers or None if buffer is tagged."""

        not_tagged_buffers = []
        nvds_batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        for nvds_frame_meta in nvds_frame_meta_iterator(nvds_batch_meta):
            savant_frame_meta = nvds_frame_meta_get_nvds_savant_frame_meta(
                nvds_frame_meta
            )
            frame_idx = savant_frame_meta.idx if savant_frame_meta else None
            frame_pts = nvds_frame_meta.buf_pts
            self.logger.debug(
                'savant_frame_meta=%s, frame_idx=%s, frame_pts=%s',
                savant_frame_meta,
                frame_idx,
                frame_pts,
            )
            frame_meta = get_source_frame_meta(self.source_id, frame_idx, frame_pts)
            if self.tag in frame_meta.tags:
                self.logger.debug('Frame PTS=%s has tag "%s"', frame_pts, self.tag)
                return None

            not_tagged_buffer: Gst.Buffer = Gst.Buffer.new()
            not_tagged_buffer.pts = frame_pts
            not_tagged_buffer.set_flags(Gst.BufferFlags.DELTA_UNIT)
            gst_buffer_add_savant_frame_meta(not_tagged_buffer, frame_idx)
            not_tagged_buffers.append(not_tagged_buffer)

        self.logger.debug(
            'Frames PTS=%s do not have tag "%s"',
            [x.pts for x in not_tagged_buffers],
            self.tag,
        )
        return not_tagged_buffers

    def push_buffer(self, pad: Gst.Pad, buffer: Gst.Buffer):
        """Push buffer to src pad."""

        self.logger.debug('Pushing buffer PTS=%s to %s', buffer.pts, pad.get_name())
        ret = pad.push(buffer)
        if ret == Gst.FlowReturn.OK:
            self.logger.debug(
                'Buffer PTS=%s successfully pushed to %s',
                buffer.pts,
                pad.get_name(),
            )
        else:
            self.logger.error(
                'Failed to push buffer PTS=%s to %s: %s',
                buffer.pts,
                pad.get_name(),
                ret,
            )
        return ret

    def on_caps(self, pad: Gst.Pad, event: Gst.Event):
        """Pass caps event to src_pad_tagged pad for negotiation purposes."""

        caps: Gst.Caps = event.parse_caps()
        self.logger.info('Caps on pad "%s" changed to %s', pad.get_name(), caps)
        self.src_pad_tagged.push_event(event)
        return Gst.PadProbeReturn.OK


# register plugin
GObject.type_register(FrameTagFilter)
__gstelementfactory__ = (
    FrameTagFilter.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    FrameTagFilter,
)
