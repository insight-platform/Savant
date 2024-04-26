from typing import List

from savant.gstreamer import GObject, Gst
from savant.utils.logging import LoggerMixin

SINK_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink',
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.new_any(),
)
SRC_PAD_TEMPLATE = Gst.PadTemplate.new(
    'src',
    Gst.PadDirection.SRC,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.new_any(),
)


class SetDts(LoggerMixin, Gst.Element):
    """Set DTS on encoded frames if not set.

    Stores buffers in a queue until the next keyframe is received, then sets
    DTS and pushes the frames from queue.
    """

    GST_PLUGIN_NAME = 'set_dts'

    __gstmetadata__ = (
        'Set DTS on encoded frames if not set.',
        'Transform',
        'Stores buffers in a queue until the next keyframe is received, then '
        'sets DTS and pushes the frames from queue.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        SINK_PAD_TEMPLATE,
        SRC_PAD_TEMPLATE,
    )

    def __init__(self):
        super().__init__()

        self.buffers: List[Gst.Buffer] = []
        self.sink_pad: Gst.Pad = Gst.Pad.new_from_template(SINK_PAD_TEMPLATE, 'sink')
        self.src_pad: Gst.Pad = Gst.Pad.new_from_template(SRC_PAD_TEMPLATE, 'src')

        self.add_pad(self.sink_pad)
        self.add_pad(self.src_pad)

        self.sink_pad.set_chain_function_full(self.handle_buffer)
        self.sink_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            self.on_pad_event,
        )

    def handle_buffer(
        self,
        sink_pad: Gst.Pad,
        element: Gst.Element,
        buffer: Gst.Buffer,
    ) -> Gst.FlowReturn:
        """Handle buffer from a sink pad."""

        self.logger.debug(
            'Received buffer PTS=%s from %s', buffer.pts, sink_pad.get_name()
        )

        if buffer.has_flags(Gst.BufferFlags.DELTA_UNIT):
            self.buffers.append(buffer)
        else:
            self.push_pending_buffers()
            buffer.dts = buffer.pts
            self.push_buffer(buffer)

        return Gst.FlowReturn.OK

    def on_pad_event(self, pad: Gst.Pad, info: Gst.PadProbeInfo):
        """Handle sink pad event."""

        event: Gst.Event = info.get_event()
        self.logger.debug('Received event %s from %s', event.type, pad.get_name())
        if event.type == Gst.EventType.EOS:
            self.logger.info('Got EOS from %s', pad.get_name())
            self.push_pending_buffers()

        return self.src_pad.push_event(event)

    def push_pending_buffers(self):
        """Push pending buffers to the src pad."""

        self.logger.debug('Pushing %s pending buffers', len(self.buffers))
        if not self.buffers:
            return
        pending_pts = [x.pts for x in self.buffers]
        pending_pts.sort()
        for x, ts in zip(self.buffers, pending_pts):
            x.dts = ts
            self.push_buffer(x)
        self.buffers = []

    def push_buffer(self, buffer: Gst.Buffer):
        """Push buffer to the src pad."""

        self.logger.debug('Pushing buffer PTS=%s', buffer.pts)
        return self.src_pad.push(buffer)


# register plugin
GObject.type_register(SetDts)
__gstelementfactory__ = (
    SetDts.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SetDts,
)
