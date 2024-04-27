import heapq
from collections import deque
from typing import Deque, List

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

        self.buffers: Deque[Gst.Buffer] = deque()
        self.pts_list: List[int] = []
        self.sink_pad: Gst.Pad = Gst.Pad.new_from_template(SINK_PAD_TEMPLATE, 'sink')
        self.src_pad: Gst.Pad = Gst.Pad.new_from_template(SRC_PAD_TEMPLATE, 'src')
        self.enabled = False

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
            'Received buffer PTS=%s DTS=%s from %s',
            buffer.pts,
            buffer.dts,
            sink_pad.get_name(),
        )
        if not self.enabled:
            return self.push_buffer(buffer)

        if buffer.has_flags(Gst.BufferFlags.DELTA_UNIT) and (
            self.buffers or buffer.dts == Gst.CLOCK_TIME_NONE
        ):
            self.logger.debug(
                'Storing buffer PTS=%s DTS=%s in a queue',
                buffer.pts,
                buffer.dts,
            )
            self.buffers.append(buffer)
            heapq.heappush(self.pts_list, buffer.pts)
            flow_ret = Gst.FlowReturn.OK

        else:
            flow_ret = self.push_pending_buffers()
            if flow_ret != Gst.FlowReturn.OK:
                return flow_ret

            if buffer.dts == Gst.CLOCK_TIME_NONE:
                self.logger.debug(
                    'Setting DTS=%s for buffer PTS=%s',
                    buffer.dts,
                    buffer.pts,
                )
                buffer.dts = buffer.pts
            flow_ret = self.push_buffer(buffer)

        return flow_ret

    def on_pad_event(self, pad: Gst.Pad, info: Gst.PadProbeInfo) -> Gst.PadProbeReturn:
        """Handle sink pad event."""

        event: Gst.Event = info.get_event()
        self.logger.debug('Received event %s from %s', event.type, pad.get_name())
        if event.type == Gst.EventType.EOS:
            self.logger.info('Got EOS from %s', pad.get_name())
            self.push_pending_buffers()

        if event.type == Gst.EventType.CAPS:
            caps: Gst.Caps = event.parse_caps()
            self.logger.debug('Got CAPS %s', caps.to_string())
            caps_name = caps[0].get_name()
            self.enabled = caps_name.startswith('video/') and not caps_name.startswith(
                'video/x-raw'
            )
            self.push_pending_buffers()

        elif event.type == Gst.EventType.SEGMENT:
            self.push_pending_buffers()

        elif event.type == Gst.EventType.STREAM_START:
            self.push_pending_buffers()

        self.src_pad.push_event(event)

        return Gst.PadProbeReturn.OK

    def push_pending_buffers(self) -> Gst.FlowReturn:
        """Push pending buffers to the src pad."""

        self.logger.debug('Pushing %s pending buffers', len(self.buffers))
        while self.buffers:
            buffer = self.buffers.popleft()
            ts = heapq.heappop(self.pts_list)
            buffer.dts = ts
            flow_ret = self.push_buffer(buffer)
            if flow_ret != Gst.FlowReturn.OK:
                return flow_ret

        return Gst.FlowReturn.OK

    def push_buffer(self, buffer: Gst.Buffer) -> Gst.FlowReturn:
        """Push buffer to the src pad."""

        self.logger.debug('Pushing buffer PTS=%s DTS=%s', buffer.pts, buffer.dts)
        flow_ret = self.src_pad.push(buffer)
        if flow_ret == Gst.FlowReturn.OK:
            self.logger.debug('Buffer PTS=%s successfully pushed', buffer.pts)
        else:
            self.logger.error('Failed to push buffer PTS=%s: %s', buffer.pts, flow_ret)

        return flow_ret


# register plugin
GObject.type_register(SetDts)
__gstelementfactory__ = (
    SetDts.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SetDts,
)
