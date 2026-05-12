"""GStreamer plugin to execute multiple user-defined Python functions
sequentially within a single element.

Can be used for metadata conversion, inference post-processing, and
other tasks.
"""

import itertools
from typing import Any, List, Optional

from savant_rs.pipeline2 import VideoPipeline

from gst_plugins.python.pyfunc_common import handle_fatal_error, init_pyfunc
from savant.base.pyfunc import BasePyFuncPlugin, PyFunc
from savant.deepstream.pygroup import NvDsPyGroupPlugin
from savant.gstreamer import GLib, GObject, Gst, GstBase  # noqa: F401
from savant.utils.log import LoggerMixin

# RGBA format is required to access the frame (pyds.get_nvds_buf_surface)
CAPS = Gst.Caps.from_string(
    'video/x-raw(memory:NVMM), '
    'format={RGBA}, '
    f'width={Gst.IntRange(range(1, GLib.MAXINT))}, '
    f'height={Gst.IntRange(range(1, GLib.MAXINT))}, '
    f'framerate={Gst.FractionRange(Gst.Fraction(0, 1), Gst.Fraction(GLib.MAXINT, 1))}'
)


class GstPluginPyGroup(LoggerMixin, GstBase.BaseTransform):
    """PyGroup GStreamer plugin."""

    GST_PLUGIN_NAME: str = 'pygroup'

    __gstmetadata__ = (
        'GStreamer element to execute multiple user-defined Python functions',
        'Transform',
        'Provides a callback to execute user-defined Python functions on every frame. '
        'Can be used for metadata conversion, inference post-processing, etc.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'sink', Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, CAPS
        ),
        Gst.PadTemplate.new('src', Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, CAPS),
        Gst.PadTemplate.new(
            'aux_src_%u', Gst.PadDirection.SRC, Gst.PadPresence.REQUEST, CAPS
        ),
    )

    __gproperties__ = {
        'elements': (
            object,
            'PyFunc elements configuration',
            'List of PyFunc element configurations '
            '(each item is a dict with module, class_name, kwargs).',
            GObject.ParamFlags.READWRITE,
        ),
        'pipeline': (
            object,
            'VideoPipeline object from savant-rs.',
            'VideoPipeline object from savant-rs.',
            GObject.ParamFlags.READWRITE,
        ),
        'gst-pipeline': (
            object,
            'GstPipeline object.',
            'GstPipeline object.',
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
        self.elements: Optional[List['PyFuncElement']] = None
        self.video_pipeline: Optional[VideoPipeline] = None
        self.gst_pipeline: Optional['GstPipeline'] = None  # noqa: F821
        self.dev_mode: bool = False
        self.max_stream_pool_size: int = 1
        # pygroup object
        self.pygroup: Optional[NvDsPyGroupPlugin] = None
        self._aux_pad_idx_gen = itertools.count()

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        if prop.name == 'elements':
            return self.elements
        if prop.name == 'pipeline':
            return self.video_pipeline
        if prop.name == 'gst-pipeline':
            return self.gst_pipeline
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
        if prop.name == 'elements':
            self.elements = value
        elif prop.name == 'pipeline':
            self.video_pipeline = value
        elif prop.name == 'gst-pipeline':
            self.gst_pipeline = value
        elif prop.name == 'stream-pool-size':
            self.max_stream_pool_size = value
        elif prop.name == 'dev-mode':
            self.dev_mode = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_start(self) -> bool:
        """Do on plugin start."""

        if not self.elements:
            return handle_fatal_error(
                self,
                self.logger,
                None,
                'Elements configuration is required for pygroup.',
                self.dev_mode,
                True,
                False,
            )

        pyfuncs: List[PyFunc] = []
        span_names: List[str] = []
        for elem in self.elements:
            kwargs_json = elem.properties.get('kwargs')

            pyfunc = init_pyfunc(
                self,
                self.logger,
                elem.module,
                elem.class_name,
                kwargs_json,
                self.dev_mode,
            )
            if pyfunc is None:
                return handle_fatal_error(
                    self,
                    self.logger,
                    None,
                    f'Failed to initialize "{elem.module}.{elem.class_name}" pyfunc in pygroup.',
                    self.dev_mode,
                    True,
                    False,
                )

            try:
                assert isinstance(pyfunc.instance, BasePyFuncPlugin), (
                    f'"{pyfunc}" should be an instance of "BasePyFuncPlugin" subclass.'
                )
                pyfunc.instance.gst_element = self
            except Exception as exc:
                return handle_fatal_error(
                    self,
                    self.logger,
                    exc,
                    f'Error validating "{elem.module}.{elem.class_name}" in pygroup.',
                    self.dev_mode,
                    True,
                    False,
                )

            pyfuncs.append(pyfunc)
            span_names.append(f'{elem.module}.{elem.class_name}')

        self.pygroup = NvDsPyGroupPlugin(pyfuncs, span_names)
        try:
            self.pygroup.gst_element = self
            return self.pygroup.on_start()
        except Exception as exc:
            return handle_fatal_error(
                self,
                self.logger,
                exc,
                'Error in on_start() call for pygroup.',
                self.dev_mode,
                True,
                False,
            )

    def do_stop(self) -> bool:
        """Do on plugin stop."""
        # pylint: disable=broad-exception-caught
        try:
            return self.pygroup.on_stop()
        except Exception as exc:
            return handle_fatal_error(
                self,
                self.logger,
                exc,
                f'Error in do_stop() call for {self.pygroup}',
                self.dev_mode,
                True,
                False,
            )

    def do_sink_event(self, event: Gst.Event) -> bool:
        """Do on sink event."""
        # pylint: disable=broad-exception-caught
        try:
            self.pygroup.on_event(event)
        except Exception as exc:
            res = handle_fatal_error(
                self,
                self.logger,
                exc,
                f'Error in do_sink_event() call for {self.pygroup}.',
                self.dev_mode,
                True,
                False,
            )
            if not res:
                return False
        return self.srcpad.push_event(event)

    def do_transform_ip(self, buffer: Gst.Buffer):
        """Transform buffer in-place function."""
        # pylint: disable=broad-exception-caught
        try:
            self.pygroup.process_buffer(buffer)
        except Exception as exc:
            return handle_fatal_error(
                self,
                self.logger,
                exc,
                f'Error in process_buffer() call for {self.pygroup}.',
                self.dev_mode,
                Gst.FlowReturn.OK,
                Gst.FlowReturn.ERROR,
            )

        return Gst.FlowReturn.OK

    def do_request_new_pad(
        self,
        templ: Gst.PadTemplate,
        name: str = None,
        caps: Gst.Caps = None,
    ):
        """Create a new pad on request."""

        pad_name = templ.name_template % next(self._aux_pad_idx_gen)
        self.logger.info('Creating auxiliary pad %s', pad_name)
        pad: Gst.Pad = Gst.Pad.new_from_template(templ, pad_name)
        if pad is None:
            self.logger.error('Failed to create pad %s', pad_name)
            return None

        self.logger.debug('Created pad %s', pad.get_name())
        self.add_pad(pad)

        return pad


# register plugin
GObject.type_register(GstPluginPyGroup)
__gstelementfactory__ = (
    GstPluginPyGroup.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    GstPluginPyGroup,
)
