from typing import Any

from savant.gstreamer import GObject, Gst, GstBase
from savant.utils.logging import LoggerMixin


class ShiftTimestamps(LoggerMixin, GstBase.BaseTransform):
    """Shift buffer timestamps."""

    GST_PLUGIN_NAME: str = 'shift_timestamps'

    __gstmetadata__ = (
        'Shift timestamps',
        'Transform',
        'Shift buffer timestamps.',
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
        'offset': (
            GObject.TYPE_LONG,
            'Shift timestamps by this value.',
            'Shift timestamps by this value.',
            0,
            GObject.G_MAXLONG,
            0,
            GObject.ParamFlags.READWRITE,
        )
    }

    def __init__(self):
        super().__init__()
        self.offset = 0
        self.set_in_place(True)

    def do_get_property(self, prop: GObject.GParamSpec):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        if prop.name == 'offset':
            return self.offset
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        if prop.name == 'offset':
            self.offset = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_transform_ip(self, buffer: Gst.Buffer):
        if buffer.pts != Gst.CLOCK_TIME_NONE:
            buffer.pts += self.offset
        if buffer.dts != Gst.CLOCK_TIME_NONE:
            buffer.dts += self.offset

        return Gst.FlowReturn.OK


# register plugin
GObject.type_register(ShiftTimestamps)
__gstelementfactory__ = (
    ShiftTimestamps.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    ShiftTimestamps,
)
