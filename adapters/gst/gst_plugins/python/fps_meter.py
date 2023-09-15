from enum import Enum
from typing import Any, Optional

from savant.gstreamer import GObject, Gst, GstBase
from savant.utils.fps_meter import FPSMeter
from savant.utils.logging import LoggerMixin


class Output(Enum):
    STDOUT = 'stdout'
    LOGGER = 'logger'


DEFAULT_OUTPUT = Output.STDOUT
DEFAULT_PERIOD_FRAMES = 1000


class FPSMeterPlugin(LoggerMixin, GstBase.BaseTransform):
    """GStreamer plugin to measure FPS."""

    GST_PLUGIN_NAME: str = 'fps_meter'

    __gstmetadata__ = (
        'Measures FPS',
        'Transform',
        'Measures FPS',
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
        # TODO: make enum
        'output': (
            str,
            'Where to dump stats',
            f'Where to dump stats. One of {", ".join(x.value for x in Output)}',
            DEFAULT_OUTPUT.value,
            GObject.ParamFlags.READWRITE,
        ),
        'period-frames': (
            int,
            'FPS measurement period',
            'FPS measurement period, in frames',
            1,
            GObject.G_MAXINT,
            DEFAULT_PERIOD_FRAMES,
            GObject.ParamFlags.READWRITE,
        ),
        'period-seconds': (
            float,
            'FPS measurement period',
            'FPS measurement period, in seconds',
            1,
            GObject.G_MAXDOUBLE,
            1,
            GObject.ParamFlags.READWRITE,
        ),
        'measure-per-file': (
            bool,
            'Measure FPS per file',
            'Measure FPS per file. FPS meter will dump statistics at the end of each file.',
            True,
            GObject.ParamFlags.READWRITE,
        ),
        'measure-per-loop': (
            bool,
            'Measure FPS per loop',
            'Measure FPS per loop. FPS meter will dump statistics at the end of each loop.',
            False,
            GObject.ParamFlags.READWRITE,
        ),
        'measurer-name': (
            str,
            'Name of the measurer',
            'Name of the measurer to indicate who is reporting measured FPS.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        # properties
        self.output: Output = DEFAULT_OUTPUT
        self.measure_per_file = True
        self.measure_per_loop = False
        self.measurer_name: Optional[str] = None

        self.location: Optional[str] = None
        self.fps_meter = FPSMeter(period_frames=DEFAULT_PERIOD_FRAMES)

        self.set_in_place(True)
        self.set_passthrough(True)

    def do_get_property(self, prop: GObject.GParamSpec):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        if prop.name == 'output':
            return self.output.value
        if prop.name == 'period-frames':
            return self.fps_meter.period_frames
        if prop.name == 'period-seconds':
            return self.fps_meter.period_seconds
        if prop.name == 'measure-per-file':
            return self.measure_per_file
        if prop.name == 'measure-per-loop':
            return self.measure_per_loop
        if prop.name == 'measurer-name':
            return self.measurer_name
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        if prop.name == 'output':
            self.output = Output(value)
        elif prop.name == 'period-frames':
            self.fps_meter.period_frames = value
        elif prop.name == 'period-seconds':
            self.fps_meter.period_seconds = value
        elif prop.name == 'measure-per-file':
            self.measure_per_file = value
        elif prop.name == 'measure-per-loop':
            self.measure_per_loop = value
        elif prop.name == 'measurer-name':
            self.measurer_name = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        self.fps_meter.start()
        return True

    def do_transform_ip(self, buf: Gst.Buffer):
        if self.fps_meter(1):
            self.dump_stats()
        return Gst.FlowReturn.OK

    def do_sink_event(self, event: Gst.Event):
        if event.type == Gst.EventType.EOS:
            self.logger.debug('Got End-Of-Stream event')
            self.fps_meter.reset_counter()
            self.dump_stats()

        elif (
            self.measure_per_file or self.measure_per_loop
        ) and event.type == Gst.EventType.TAG:
            tag_list: Gst.TagList = event.parse_tag()
            has_location, location = tag_list.get_string(Gst.TAG_LOCATION)
            if has_location:
                if self.location and (
                    self.measure_per_loop or self.location != location
                ):
                    self.dump_stats()
                    self.fps_meter.reset_counter()
                self.logger.debug('Set location to %s', location)
                self.location = location

        # Cannot use `super()` since it is `self`
        return GstBase.BaseTransform.do_sink_event(self, event)

    def dump_stats(self):
        message = self.fps_meter.message
        if self.measurer_name is not None:
            message = f'{self.measurer_name}: {message}'
        if self.location:
            message += f' Source location: {self.location}.'
        if self.output == Output.STDOUT:
            print(message)
        else:
            self.logger.info(message)


# register plugin
GObject.type_register(FPSMeterPlugin)
__gstelementfactory__ = (
    FPSMeterPlugin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    FPSMeterPlugin,
)
