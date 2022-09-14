import logging
import os

from savant.gstreamer import GObject, Gst, GstBase
from savant.gstreamer.utils import LoggerMixin

logging.basicConfig(
    level=os.environ.get('LOGLEVEL', 'INFO'),
    format='%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] %(message)s',
)


class Logger(LoggerMixin, GstBase.BaseTransform):
    """Logs buffers and events for debugging purpose."""

    GST_PLUGIN_NAME: str = 'logger'

    __gstmetadata__ = (
        'Logger',
        'Transform',
        'Logs buffers and events for debugging purpose',
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
        'buffer-content': (
            bool,
            'Log buffer content',
            'Log buffer content',
            False,
            GObject.ParamFlags.READWRITE,
        ),
        'buffer-flags': (
            bool,
            'Log buffer flags',
            'Log buffer flags',
            False,
            GObject.ParamFlags.READWRITE,
        ),
        'buffer': (
            bool,
            'Log buffers',
            'Log buffers',
            True,
            GObject.ParamFlags.READWRITE,
        ),
        'event': (
            bool,
            'Log events',
            'Log events',
            True,
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        self._buffer_content = False
        self._buffer_flags = False
        self._buffer = True
        self._event = True
        self.set_in_place(True)
        self.set_passthrough(True)

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: property parameters
        """
        if prop.name == 'buffer-content':
            return self._buffer_content
        if prop.name == 'buffer-flags':
            return self._buffer_flags
        if prop.name == 'buffer':
            return self._buffer
        if prop.name == 'event':
            return self._event
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop, value):
        """Gst plugin set property function.

        :param prop: property parameters
        :param value: new value for param, type dependents on param
        """
        if prop.name == 'buffer-content':
            self._buffer_content = value
        elif prop.name == 'buffer-flags':
            self._buffer_flags = value
        elif prop.name == 'buffer':
            self._buffer = value
        elif prop.name == 'event':
            self._event = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_transform_ip(self, buffer: Gst.Buffer):
        if self._buffer:
            log_line = 'Got buffer. PTS: %s. DTS: %s. Duration: %s. Size: %s.'
            args = [
                buffer.pts,
                buffer.dts,
                buffer.duration,
                buffer.get_size(),
            ]
            if self._buffer_flags:
                log_line += ' Flags: %s.'
                flags: Gst.BufferFlags = buffer.get_flags()
                args.append(
                    ', '.join(flags.value_names) if flags is not None else flags
                )
            if self._buffer_content:
                log_line += ' Content: %s.'
                args.append(buffer.extract_dup(0, buffer.get_size()))
            self.logger.info(log_line, *args)
        return Gst.FlowReturn.OK

    def do_src_event(self, event: Gst.Event):
        if self._event:
            self.logger.info('Got src event. %s', _parse_event(event))
        # Cannot use `super()` since it is `self`
        return GstBase.BaseTransform.do_src_event(self, event)

    def do_sink_event(self, event: Gst.Event):
        if self._event:
            self.logger.info('Got sink event. %s', _parse_event(event))
        # Cannot use `super()` since it is `self`
        return GstBase.BaseTransform.do_sink_event(self, event)


def _parse_event(event: Gst.Event):
    value = None
    if event.type == Gst.EventType.TAG:
        value = event.parse_tag().to_string()
    elif event.type == Gst.EventType.CAPS:
        value = event.parse_caps().to_string()
    elif event.type == Gst.EventType.STREAM_START:
        value = event.parse_stream_start()
    elif event.type == Gst.EventType.STREAM_GROUP_DONE:
        value = event.parse_stream_group_done()
    elif event.type == Gst.EventType.SEGMENT:
        x: Gst.Segment = event.parse_segment()
        value = ', '.join(
            [
                f'format={x.format.value_name}',
                f'duration={x.duration}',
                f'start={x.start}',
                f'stop={x.stop}',
                f'position={x.position}',
                f'time={x.time}',
                f'base={x.base}',
                f'offset={x.offset}',
                f'flags={x.flags.value_names}',
            ]
        )
    elif event.type == Gst.EventType.LATENCY:
        value = event.parse_latency()

    message = f'Type: {event.type.value_name}'
    if value is not None:
        message += f'. Value: {value}'
    struct = event.get_structure()
    message += f'. Structure: {struct}'
    return message


# register plugin
GObject.type_register(Logger)
__gstelementfactory__ = (
    Logger.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    Logger,
)
