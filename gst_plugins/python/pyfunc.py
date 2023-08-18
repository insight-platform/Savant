"""GStreamer plugin to execute user-defined Python functions.

Can be used for metadata conversion, inference post-processing, and
other tasks.
"""
import json
from typing import Any, Optional

from savant_rs.pipeline import VideoPipeline

from savant.base.pyfunc import BasePyFuncPlugin, PyFunc
from savant.gstreamer import GLib, GObject, Gst, GstBase  # noqa: F401
from savant.utils.logging import LoggerMixin

# RGBA format is required to access the frame (pyds.get_nvds_buf_surface)
CAPS = Gst.Caps.from_string(
    'video/x-raw(memory:NVMM), '
    'format={RGBA}, '
    f'width={Gst.IntRange(range(1, GLib.MAXINT))}, '
    f'height={Gst.IntRange(range(1, GLib.MAXINT))}, '
    f'framerate={Gst.FractionRange(Gst.Fraction(0, 1), Gst.Fraction(GLib.MAXINT, 1))}'
)


class GstPluginPyFunc(LoggerMixin, GstBase.BaseTransform):
    """PyFunc GStreamer plugin."""

    GST_PLUGIN_NAME: str = 'pyfunc'

    __gstmetadata__ = (
        'GStreamer element to execute user-defined Python function',
        'Transform',
        'Provides a callback to execute user-defined Python functions on every frame. '
        'Can be used for metadata conversion, inference post-processing, etc.',
        'Den Medyantsev <medyantsev_dv@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'sink', Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, CAPS
        ),
        Gst.PadTemplate.new('src', Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, CAPS),
    )

    __gproperties__ = {
        'module': (
            str,
            'Python module',
            'Python module name to import or module path.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'class': (
            str,
            'Python class name',
            'Python class name to instantiate.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'kwargs': (
            str,
            'Keyword arguments for class initialization',
            'Keyword argument for Python class initialization.',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'pipeline': (
            object,
            'VideoPipeline object from savant-rs.',
            'VideoPipeline object from savant-rs.',
            GObject.ParamFlags.READWRITE,
        ),
    }

    def __init__(self):
        super().__init__()
        # properties
        self.module: Optional[str] = None
        self.class_name: Optional[str] = None
        self.kwargs: Optional[str] = None
        self.video_pipeline: Optional[VideoPipeline] = None
        # pyfunc object
        self.pyfunc: Optional[BasePyFuncPlugin] = None

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        if prop.name == 'module':
            return self.module
        if prop.name == 'class':
            return self.class_name
        if prop.name == 'kwargs':
            return self.kwargs
        if prop.name == 'pipeline':
            return self.video_pipeline
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        if prop.name == 'module':
            self.module = value
        elif prop.name == 'class':
            self.class_name = value
        elif prop.name == 'kwargs':
            self.kwargs = value
        elif prop.name == 'pipeline':
            self.video_pipeline = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        """Do on plugin start."""
        self.pyfunc = PyFunc(
            module=self.module,
            class_name=self.class_name,
            kwargs=json.loads(self.kwargs) if self.kwargs else None,
        ).instance
        assert isinstance(
            self.pyfunc, BasePyFuncPlugin
        ), f'"{self.pyfunc}" should be an instance of "BasePyFuncPlugin" subclass.'
        self.pyfunc.gst_element = self
        return self.pyfunc.on_start()

    def do_stop(self):
        """Do on plugin stop."""
        return self.pyfunc.on_stop()

    def do_sink_event(self, event: Gst.Event) -> bool:
        """Do on sink event."""
        self.pyfunc.on_event(event)
        return self.srcpad.push_event(event)

    def do_transform_ip(self, buffer: Gst.Buffer):
        """Transform buffer in-place function."""
        try:
            self.pyfunc.process_buffer(buffer)
        except:
            self.logger.exception('Failed to process buffer/frame.')
            return Gst.FlowReturn.ERROR
        return Gst.FlowReturn.OK


# register plugin
GObject.type_register(GstPluginPyFunc)
__gstelementfactory__ = (
    GstPluginPyFunc.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    GstPluginPyFunc,
)
