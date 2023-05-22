from typing import List

from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.codecs import Codec
from savant.utils.logging import LoggerMixin

CAPS = Gst.Caps.from_string(
    ';'.join(x.value.caps_with_params for x in [Codec.H264, Codec.HEVC])
)
SINK_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink',
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    CAPS,
)
SRC_PAD_TEMPLATE = Gst.PadTemplate.new(
    'src',
    Gst.PadDirection.SRC,
    Gst.PadPresence.ALWAYS,
    CAPS,
)


class CalculateDts(LoggerMixin, GstBase.BaseTransform):
    """Calculate DTS for encoded frames based on their PTS."""

    GST_PLUGIN_NAME: str = 'calculate_dts'

    __gstmetadata__ = (
        'Calculate DTS for frames',
        'Transform',
        'Calculate DTS for encoded frames based on their PTS.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = SINK_PAD_TEMPLATE, SRC_PAD_TEMPLATE

    def __init__(self):
        super().__init__()
        self.queue: List[Gst.Buffer] = []

    def do_submit_input_buffer(self, is_discont: bool, buffer: Gst.Buffer):
        self.logger.debug('Got buffer with PTS %s and DTS %s.', buffer.pts, buffer.dts)

        if buffer.dts != Gst.CLOCK_TIME_NONE and not self.queue:
            return self.push_buffer(buffer)

        if buffer.has_flags(Gst.BufferFlags.DELTA_UNIT):
            self.queue.append(buffer)
            return Gst.FlowReturn.OK

        ret = self.push_buffers()
        if ret == Gst.FlowReturn.OK and buffer.dts != Gst.CLOCK_TIME_NONE:
            self.queue = []
            return self.push_buffer(buffer)

        self.queue = [buffer]

        return ret

    def push_buffers(self) -> Gst.FlowReturn:
        self.logger.debug('Pushing %s buffers.', len(self.queue))
        dts_list = sorted(x.pts for x in self.queue)
        for buffer in self.queue:
            if buffer.pts != Gst.CLOCK_TIME_NONE:
                buffer.dts = dts_list.pop(0)
            ret = self.push_buffer(buffer)
            if ret != Gst.FlowReturn.OK:
                return ret

        return Gst.FlowReturn.OK

    def push_buffer(self, buffer: Gst.Buffer) -> Gst.FlowReturn:
        self.logger.debug(
            'Pushing buffer with PTS %s and DTS %s.',
            buffer.pts,
            buffer.dts,
        )
        return self.srcpad.push(buffer)

    def do_sink_event(self, event: Gst.Event):
        if event.type == Gst.EventType.EOS:
            self.logger.info('Got EOS.')
            self.push_buffers()
            self.queue = []
        return GstBase.BaseTransform.do_sink_event(self, event)


# register plugin
GObject.type_register(CalculateDts)
__gstelementfactory__ = (
    CalculateDts.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    CalculateDts,
)
