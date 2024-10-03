from typing import Any, Optional

from savant_rs.pipeline2 import VideoPipeline

# from savant.deepstream.ingress_converter import BaseIngressConverter
from savant.gstreamer import GLib, GObject, Gst, GstBase  # noqa: F401
from savant.gstreamer.utils import on_pad_event
from savant.utils.logging import LoggerMixin

# RGBA format is required to access the frame (pyds.get_nvds_buf_surface)
CAPS = Gst.Caps.from_string(
    'video/x-raw(memory:NVMM), '
    'format={RGBA}, '
    f'width={Gst.IntRange(range(1, GLib.MAXINT))}, '
    f'height={Gst.IntRange(range(1, GLib.MAXINT))}, '
    f'framerate={Gst.FractionRange(Gst.Fraction(0, 1), Gst.Fraction(GLib.MAXINT, 1))}'
)


class IngressConverterBin(LoggerMixin, Gst.Bin):
    """IngressConverter GStreamer plugin."""

    GST_PLUGIN_NAME: str = 'ingress_converter_bin'

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
        self._converter: Optional['BaseIngressConverter'] = None
        self._video_pipeline: Optional[VideoPipeline] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None

        self._frame_source: Optional[Gst.Element] = None
        self._capsfilter: Optional[Gst.Element] = None
        self._converter_elem: Gst.Element = Gst.ElementFactory.make('ingress_converter')
        self.add(self._converter_elem)

        self._srcpad: Gst.GhostPad = Gst.GhostPad.new(
            'src', self._converter_elem.get_static_pad('src')
        )
        self.add_pad(self._srcpad)

        self._sinkpad: Gst.GhostPad = Gst.GhostPad.new(
            'sink', self._converter_elem.get_static_pad('sink_orig')
        )
        self._sinkpad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.CAPS: self.on_sink_pad_caps},
        )
        self.add_pad(self._sinkpad)

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        self.logger.info('Getting property %s', prop.name)
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
        self.logger.info('Setting property %s to %s', prop.name, value)
        if prop.name == 'source-id':
            self._source_id = value
            self._converter_elem.set_property('source-id', value)
            return
        if prop.name == 'converter':
            self._converter = value
            self._converter_elem.set_property('converter', value)
            return
        elif prop.name == 'pipeline':
            self._video_pipeline = value
            self._converter_elem.set_property('pipeline', value)
            return
        else:
            raise AttributeError(f'Unknown property {prop.name}.')

    def on_sink_pad_caps(self, pad: Gst.Pad, event: Gst.Event):
        caps: Gst.Caps = event.parse_caps()
        self.logger.info('Got caps: %s', caps)

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
        self.logger.info('New resolution: %s', new_resolution)
        new_caps: Gst.Caps = caps.copy()
        if new_resolution is None or new_resolution == (width, height):
            self._width = width
            self._height = height
        else:
            self._width, self._height = new_resolution
            new_caps.set_value('width', self._width)
            new_caps.set_value('height', self._height)

        self.logger.info('New caps: %s', new_caps.to_string())

        self._capsfilter = Gst.ElementFactory.make('capsfilter')
        self._capsfilter.set_property('caps', new_caps)
        self.add(self._capsfilter)
        self.logger.info('Created capsfilter element')

        assert (
            self._capsfilter.get_static_pad('src').link(
                self._converter_elem.get_static_pad('sink_conv')
            )
            == Gst.PadLinkReturn.OK
        )
        self.logger.info('Linked capsfilter to converter')

        self._frame_source = Gst.ElementFactory.make('nvvideotestsrc')
        self.add(self._frame_source)
        self.logger.info('Created frame source element')

        assert self._frame_source.link(self._capsfilter)
        self.logger.info('Linked frame source to capsfilter')

        self._converter_elem.sync_state_with_parent()
        self.logger.info('Synced converter with parent')
        self._capsfilter.sync_state_with_parent()
        self.logger.info('Synced capsfilter with parent')
        self._frame_source.sync_state_with_parent()
        self.logger.info('Synced frame source with parent')

        return Gst.PadProbeReturn.OK


# register plugin
GObject.type_register(IngressConverterBin)
__gstelementfactory__ = (
    IngressConverterBin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    IngressConverterBin,
)
