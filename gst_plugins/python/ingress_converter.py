"""GStreamer plugin to execute user-defined ingress converter.

Can be used for modifying frame resolution, content, etc.
"""

from typing import Any, Optional, Tuple

from pynvbufsurfacegenerator import NvBufSurfaceGenerator
from savant_rs.pipeline2 import VideoPipeline

from savant.deepstream.ingress_converter import BaseIngressConverter
from savant.deepstream.opencv_utils import nvds_to_gpu_mat
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


class IngressConverter(LoggerMixin, GstBase.BaseTransform):
    """IngressConverter GStreamer plugin."""

    GST_PLUGIN_NAME: str = 'ingress_converter'

    __gstmetadata__ = (
        'GStreamer plugin to execute user-defined ingress converter',
        'Transform',
        'Provides a callback to execute user-defined ingress converter on every frame. '
        'Can be used for modifying frame resolution, content, etc.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'sink',
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            CAPS,
        ),
        Gst.PadTemplate.new(
            'src',
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            CAPS,
        ),
    )

    __gproperties__ = {
        'source-id': (
            str,
            'Source ID',
            'Source ID',
            None,
            GObject.ParamFlags.READWRITE,
        ),
        'converter': (
            object,
            'Ingress converter instance',
            'Ingress converter instance',
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
        self._source_id: Optional[str] = None
        self._converter: Optional[BaseIngressConverter] = None
        self._video_pipeline: Optional[VideoPipeline] = None
        self._surface_generator: Optional[NvBufSurfaceGenerator] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        # TODO: depends on arch
        self._memory_type = int(pyds.NVBUF_MEM_DEFAULT)

        self.set_in_place(False)
        self.set_passthrough(False)

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        if prop.name == 'source-id':
            return self._source_id
        if prop.name == 'converter':
            return self._converter
        if prop.name == 'pipeline':
            return self._video_pipeline
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        if prop.name == 'source-id':
            self._source_id = value
        if prop.name == 'converter':
            self._converter = value
        elif prop.name == 'pipeline':
            self._video_pipeline = value
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def do_sink_event(self, event: Gst.Event) -> bool:
        """Do on sink event."""

        if event.type == Gst.EventType.CAPS:
            caps: Gst.Caps = event.parse_caps()
            self.logger.info('Got caps: %s', caps.to_string())
            structure: Gst.Structure = caps.get_structure(0)
            has_field, width = structure.get_int('width')
            if not has_field:
                self.logger.error('Failed to get width from caps.')
                return False
            has_field, height = structure.get_int('height')
            if not has_field:
                self.logger.error('Failed to get height from caps.')
                return False

            new_resolution = self._converter.on_stream_start(
                source_id=self._source_id,
                width=width,
                height=height,
            )
            if new_resolution is None or new_resolution == (width, height):
                new_caps = caps
                new_event = event
                self._width = width
                self._height = height
            else:
                self._width, self._height = new_resolution
                new_caps: Gst.Caps = caps.copy()
                new_caps.set_value('width', self._width)
                new_caps.set_value('height', self._height)
                self.logger.info('New caps: %s', new_caps.to_string())
                new_event: Gst.Event = Gst.Event.new_caps(new_caps)
            self._surface_generator = NvBufSurfaceGenerator(
                new_caps,
                0,  # TODO: get gpu_id
                self._memory_type,
            )

            return self.srcpad.push_event(new_event)

        return self.srcpad.push_event(event)

    def do_prepare_output_buffer(
        self,
        inbuf: Gst.Buffer,
    ) -> Tuple[Gst.FlowReturn, Gst.Buffer]:
        outbuf: Gst.Buffer = Gst.Buffer.new()
        self._surface_generator.create_surface(outbuf)

        with nvds_to_gpu_mat(inbuf, batch_id=0) as in_mat:
            with nvds_to_gpu_mat(outbuf, batch_id=0) as out_mat:
                # TODO: pass VideoFrame meta
                res_mat = self.converter.convert(self.source_id, None, in_mat)
                # TODO: check resolution
                res_mat.copyTo(out_mat)

        return Gst.FlowReturn.OK, outbuf


# register plugin
GObject.type_register(IngressConverter)
__gstelementfactory__ = (
    IngressConverter.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    IngressConverter,
)
