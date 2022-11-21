"""PyFuncBin."""
from typing import Any

from savant.gstreamer import Gst, GObject, GLib  # noqa: F401
from savant.gst_plugins.python.pyfunc import PROPS as PYFUNC_PROPS


class PyFuncBin(Gst.Bin):
    """GStreamer Bin to draw using base or custom implementation of DrawBin
    class."""

    GST_PLUGIN_NAME = 'pyfuncbin'

    __gstmetadata__ = (
        'PyFunc Bin',
        'Bin',
        'Wraps PyFunc for caps negotiation',
        'Den Medyantsev <medyantsev_dv@bw-sw.com>',
    )

    __gsttemplates__ = (
        # only caps = any (not specific nvvideoconvert or pyfunc caps)
        # solves the problem with memory leak on jetson
        Gst.PadTemplate.new(
            'sink', Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, Gst.Caps.new_any()
        ),
        Gst.PadTemplate.new(
            'src', Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, Gst.Caps.new_any()
        ),
    )

    __gproperties__ = PYFUNC_PROPS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._queue: Gst.Element = Gst.ElementFactory.make('queue', 'pyfunc_queue')
        self.add(self._queue)

        self._conv: Gst.Element = Gst.ElementFactory.make(
            'nvvideoconvert', 'pyfunc_conv'
        )
        self.add(self._conv)

        self._conv_queue: Gst.Element = Gst.ElementFactory.make(
            'queue', 'pyfunc_conv_queue'
        )
        self.add(self._conv_queue)

        self._pyfunc: Gst.Element = Gst.ElementFactory.make('pyfunc', 'pyfunc_pyfunc')
        self.add(self._pyfunc)

        assert self._queue.link(self._conv)
        assert self._conv.link(self._conv_queue)
        assert self._conv_queue.link(self._pyfunc)

        queue_sink_pad = self._queue.get_static_pad('sink')
        sink_pad = Gst.GhostPad.new_from_template(
            'sink', queue_sink_pad, self.__gsttemplates__[0]
        )
        assert sink_pad.set_active(True)
        self.add_pad(sink_pad)

        pyfunc_src_pad = self._pyfunc.get_static_pad('src')
        src_pad = Gst.GhostPad.new_from_template(
            'src', pyfunc_src_pad, self.__gsttemplates__[1]
        )
        assert src_pad.set_active(True)
        self.add_pad(src_pad)

    def do_get_property(self, prop):
        """Gst plugin get property function.

        :param prop: structure that encapsulates the metadata
            required to specify parameters
        """
        if prop.name in PYFUNC_PROPS:
            return self._pyfunc.get_property(prop.name)
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        if prop.name in PYFUNC_PROPS:
            self._pyfunc.set_property(prop.name, value)
        else:
            raise AttributeError(f'Unknown property {prop.name}.')


# register plugin
GObject.type_register(PyFuncBin)
__gstelementfactory__ = (
    PyFuncBin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    PyFuncBin,
)
