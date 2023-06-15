from collections import deque
from threading import Lock
from typing import Deque, Optional, Tuple

from gst_plugins.python.frame_tag_filter_common import parse_stream_part_event
from savant.gstreamer import GObject, Gst
from savant.gstreamer.utils import on_pad_event
from savant.utils.logging import LoggerMixin

SINK_TAGGED_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink_tagged',
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.new_any(),
)
SINK_NOT_TAGGED_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink_not_tagged',
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


class FrameTagFunnel(LoggerMixin, Gst.Element):
    """Frame tag funnel.

    Funnels frames after "frame_tag_filter" element back to
    a single stream with the original order.
    """

    GST_PLUGIN_NAME = 'frame_tag_funnel'

    __gstmetadata__ = (
        'Frame tag funnel',
        'Muxer',
        'Funnels frames after "frame_tag_filter" element back to '
        'a single stream with the original order.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        SINK_TAGGED_PAD_TEMPLATE,
        SINK_NOT_TAGGED_PAD_TEMPLATE,
        SRC_PAD_TEMPLATE,
    )

    def __init__(self):
        super().__init__()

        self.parts_deque: Deque[Tuple[Gst.Pad, int]] = deque()
        self.buffers: Deque[Deque[Gst.Buffer]] = deque()
        self.current_pad: Optional[Gst.Pad] = None
        self.stream_lock = Lock()

        self.sink_pad_tagged: Gst.Pad = Gst.Pad.new_from_template(
            SINK_TAGGED_PAD_TEMPLATE, 'sink_tagged'
        )
        self.sink_pad_not_tagged: Gst.Pad = Gst.Pad.new_from_template(
            SINK_NOT_TAGGED_PAD_TEMPLATE, 'sink_not_tagged'
        )
        self.tagged_eos = False
        self.not_tagged_eos = False
        self.src_pad: Gst.Pad = Gst.Pad.new_from_template(SRC_PAD_TEMPLATE, 'src')

        self.add_pad(self.sink_pad_tagged)
        self.add_pad(self.sink_pad_not_tagged)
        self.add_pad(self.src_pad)

        self.sink_pad_tagged.set_chain_function_full(self.handle_buffer)
        self.sink_pad_not_tagged.set_chain_function_full(self.handle_buffer)

        event_handlers = {
            Gst.EventType.EOS: self.on_eos,
            Gst.EventType.CUSTOM_DOWNSTREAM: self.on_custom_event,
        }
        self.sink_pad_not_tagged.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            event_handlers,
        )
        self.sink_pad_tagged.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            event_handlers,
        )

    def handle_buffer(
        self,
        sink_pad: Gst.Pad,
        element: Gst.Element,
        buffer: Gst.Buffer,
    ) -> Gst.FlowReturn:
        """Handle buffer from a sink pad.

        Either pass the buffer to the src pad or store it in the buffer queue
        for keeping the original order.
        """

        self.logger.debug(
            'Received buffer PTS=%s from %s', buffer.pts, sink_pad.get_name()
        )
        with self.stream_lock:
            if sink_pad == self.current_pad:
                return self.push_buffer(buffer)
            else:
                self.buffers[-1].append(buffer)
                return Gst.FlowReturn.OK

    def on_eos(self, pad: Gst.Pad, event: Gst.Event):
        """Handle EOS event from a sink pad.

        Pass EOS to the src pad only when received from both sink pads.
        """

        self.logger.info('Got EOS from %s', pad.get_name())

        with self.stream_lock:
            if pad == self.sink_pad_tagged:
                self.tagged_eos = True
            else:
                self.not_tagged_eos = True

            if self.not_tagged_eos and self.tagged_eos:
                self.logger.info('Pass EOS from %s', pad.get_name())
                while self.buffers:
                    for buffer in self.buffers.popleft():
                        self.push_buffer(buffer)

                return Gst.PadProbeReturn.PASS
            else:
                self.logger.info('Drop EOS from %s', pad.get_name())
                return Gst.PadProbeReturn.DROP

    def on_custom_event(self, sink_pad: Gst.Pad, event: Gst.Event):
        """Handle stream-part event from a sink pad."""

        self.logger.debug('Got CUSTOM_DOWNSTREAM event from %s', sink_pad.get_name())
        parsed_event = parse_stream_part_event(event)
        if parsed_event is None:
            return Gst.PadProbeReturn.PASS

        part_id, tagged = parsed_event
        branch_pad = self.sink_pad_tagged if tagged else self.sink_pad_not_tagged
        part = (branch_pad, part_id)

        with self.stream_lock:
            if self.parts_deque and self.parts_deque[0] == part:
                self.parts_deque.popleft()
                for buffer in self.buffers.popleft():
                    self.push_buffer(buffer)
                if not self.parts_deque or self.parts_deque[0][0] != branch_pad:
                    self.current_pad = branch_pad
                    self.logger.debug(
                        'Switched current pad to %s', self.current_pad.get_name()
                    )

            else:
                if sink_pad == self.current_pad and self.parts_deque:
                    self.current_pad = None
                self.parts_deque.append(part)
                self.buffers.append(deque())

        return Gst.PadProbeReturn.DROP

    def push_buffer(self, buffer: Gst.Buffer):
        """Push buffer to the src pad."""

        self.logger.debug('Pushing buffer PTS=%s', buffer.pts)
        return self.src_pad.push(buffer)


# register plugin
GObject.type_register(FrameTagFunnel)
__gstelementfactory__ = (
    FrameTagFunnel.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    FrameTagFunnel,
)
