"""GStreamer plugin to execute user-defined Python functions.

Can be used for metadata conversion, inference post-processing, and
other tasks.
"""
import json
from typing import Any, Optional

from savant_rs.pipeline2 import VideoPipeline

from savant.base.pyfunc import BasePyFuncPlugin, PyFunc, PyFuncNoopCallException
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
        'stream-pool-size': (
            int,
            'Max stream pool size',
            'Max stream pool size',
            1,
            GLib.MAXINT,
            1,
            GObject.ParamFlags.READWRITE,
        ),
        'dev-mode': (
            bool,
            'Dev mode flag',
            (
                'Whether to monitor source file changes at runtime'
                ' and reload the pyfunc objects when necessary.'
            ),
            False,
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
        self.dev_mode: bool = False
        self.max_stream_pool_size: int = 1
        # pyfunc object
        self.pyfunc: Optional[PyFunc] = None

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
        if prop.name == 'stream-pool-size':
            return self.max_stream_pool_size
        if prop.name == 'dev-mode':
            return self.dev_mode
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
        elif prop.name == 'stream-pool-size':
            self.max_stream_pool_size = value
        elif prop.name == 'dev-mode':
            self.dev_mode = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self):
        """Do on plugin start."""
        self.pyfunc = PyFunc(
            module=self.module,
            class_name=self.class_name,
            kwargs=json.loads(self.kwargs) if self.kwargs else None,
            dev_mode=self.dev_mode,
        )

        self.pyfunc.load_user_code()

        assert isinstance(
            self.pyfunc.instance, BasePyFuncPlugin
        ), f'"{self.pyfunc}" should be an instance of "BasePyFuncPlugin" subclass.'
        self.pyfunc.instance.gst_element = self

        try:
            return self.pyfunc.instance.on_start()
        except Exception as exc:
            return self.handle_error(exc, 'Error in pyfunc do_start()', True, False)

    def do_stop(self):
        """Do on plugin stop."""
        try:
            return self.pyfunc.instance.on_stop()
        except Exception as exc:
            return self.handle_error(exc, 'Error in pyfunc do_stop()', True, False)

    def do_sink_event(self, event: Gst.Event) -> bool:
        """Do on sink event."""
        try:
            self.pyfunc.instance.on_event(event)
        except Exception as exc:
            self.handle_error(exc, 'Error in pyfunc do_sink_event()', True, False)
        return self.srcpad.push_event(event)

    def do_transform_ip(self, buffer: Gst.Buffer):
        """Transform buffer in-place function."""
        try:
            self.pyfunc.instance.process_buffer(buffer)
        except Exception as exc:
            return self.handle_error(exc, 'Failed to process buffer/frame.')

        return Gst.FlowReturn.OK

    def handle_error(
        self, exc, msg, return_ok=Gst.FlowReturn.OK, return_err=Gst.FlowReturn.ERROR
    ):
        if self.dev_mode:
            if not isinstance(exc, PyFuncNoopCallException):
                self.logger.exception(msg)
            return return_ok
        self.logger.exception(msg)
        return return_err


# register plugin
GObject.type_register(GstPluginPyFunc)
__gstelementfactory__ = (
    GstPluginPyFunc.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    GstPluginPyFunc,
)
