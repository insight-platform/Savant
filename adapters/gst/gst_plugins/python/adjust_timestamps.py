from typing import Any

from savant.gstreamer import GObject, Gst, GstBase
from savant.utils.logging import LoggerMixin


class AdjustTimestamps(LoggerMixin, GstBase.BaseTransform):
    """Adjust buffer timestamps and segments to be monotonic."""

    GST_PLUGIN_NAME: str = 'adjust_timestamps'

    __gstmetadata__ = (
        'Adjust timestamps',
        'Transform',
        'Adjust buffer timestamps and segments to be monotonic.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'src',
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        ),
        Gst.PadTemplate.new(
            'sink',
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.new_any(),
        ),
    )

    __gproperties__ = {
        'adjust-first-frame': (
            bool,
            'Adjust timestamp for the first frame.',
            'Adjust timestamp for the first frame.',
            False,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        self.offset = 0
        self.max_pts = 0
        self.max_dts = 0
        self.adjust_first_frame = False
        self.new_segment = True
        self.set_in_place(True)

    def do_get_property(self, prop: GObject.GParamSpec):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        if prop.name == 'adjust-first-frame':
            return self.adjust_first_frame
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        if prop.name == 'adjust-first-frame':
            self.adjust_first_frame = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_transform_ip(self, buffer: Gst.Buffer):
        if self.new_segment:
            # Calculating offset only at the beginning of the segment
            self.update_offset(buffer)
            self.new_segment = False
        self.update_timestamps(buffer)

        return Gst.FlowReturn.OK

    def update_offset(self, buffer: Gst.Buffer):
        """Calculate new offset base on the expected timestamps."""

        self.logger.debug(
            'Calculating delta for buffer with PTS %s, DTS %s and duration %s',
            buffer.pts,
            buffer.dts,
            buffer.duration,
        )
        if self.adjust_first_frame:
            current_running_time = self.get_clock().get_time() - self.get_base_time()
            self.max_pts = max(self.max_pts, current_running_time)
            self.max_dts = max(self.max_dts, current_running_time)
        delta = 0
        if buffer.dts != Gst.CLOCK_TIME_NONE and buffer.dts < self.max_dts:
            self.logger.info('Buffer DTS is %s, expected: %s', buffer.dts, self.max_dts)
            delta = self.max_dts - buffer.dts

        if buffer.pts != Gst.CLOCK_TIME_NONE:
            if delta > 0:
                delta = max(delta, self.max_pts - buffer.pts)
            elif buffer.dts == Gst.CLOCK_TIME_NONE and buffer.pts < self.max_pts:
                self.logger.info(
                    'Buffer PTS is %s, expected: %s', buffer.pts, self.max_pts
                )
                delta = self.max_pts - buffer.pts

        if delta > 0:
            new_offset = self.offset + delta
            self.logger.info(
                'Increasing offset by %s: from %s to %s', delta, self.offset, new_offset
            )
            self.offset = new_offset

        self.max_dts = 0
        self.max_pts = 0

    def update_timestamps(self, buffer: Gst.Buffer):
        """Update timestamps of the buffer and save maximums."""

        duration = 0 if buffer.duration == Gst.CLOCK_TIME_NONE else buffer.duration
        if buffer.dts != Gst.CLOCK_TIME_NONE:
            self.max_dts = max(self.max_dts, buffer.dts + duration)
            if self.offset > 0:
                last_dts = buffer.dts
                buffer.dts += self.offset
                self.logger.debug(
                    'Buffer DTS updated from %s to %s', last_dts, buffer.dts
                )
        if buffer.pts != Gst.CLOCK_TIME_NONE:
            self.max_pts = max(self.max_pts, buffer.pts + duration)
            if self.offset > 0:
                last_pts = buffer.pts
                buffer.pts += self.offset
                self.logger.debug(
                    'Buffer PTS updated from %s to %s', last_pts, buffer.pts
                )

        return Gst.FlowReturn.OK

    def do_sink_event(self, event: Gst.Event):
        if event.type != Gst.EventType.SEGMENT:
            # Cannot use `super()` since it is `self`
            return GstBase.BaseTransform.do_sink_event(self, event)

        segment: Gst.Segment = event.parse_segment()
        # Sink-elements skip buffers outside of segment
        segment.start = 0
        segment.stop = Gst.CLOCK_TIME_NONE
        self.new_segment = True
        self.logger.debug('Expand segment')
        self.srcpad.push_event(Gst.Event.new_segment(segment))
        return True


# register plugin
GObject.type_register(AdjustTimestamps)
__gstelementfactory__ = (
    AdjustTimestamps.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    AdjustTimestamps,
)
