"""DrawBin element."""
from pathlib import Path
from typing import Any, Optional
import shutil

from savant.gstreamer import Gst, GObject  # noqa: F401
from savant.utils.platform import is_aarch64


CAPS = Gst.Caps.from_string('video/x-raw(memory:NVMM), format=RGBA')


class DrawBin(Gst.Bin):
    """GStreamer Bin to draw using base or custom implementation of DrawBin
    class."""

    GST_PLUGIN_NAME = 'drawbin'

    __gstmetadata__ = (
        'Draw Bin',
        'Bin',
        'Draw text, lines, bounding boxes',
        'Den Medyantsev <medyantsev_dv@bw-sw.com>',
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            'sink', Gst.PadDirection.SINK, Gst.PadPresence.ALWAYS, CAPS
        ),
        Gst.PadTemplate.new('src', Gst.PadDirection.SRC, Gst.PadPresence.ALWAYS, CAPS),
    )

    __gproperties__ = {
        'location': (
            str,
            'Location of the file to write or "display"',
            'Location of the file to write (filesink/multifilesink) '
            'or "display" (nveglglessink).',
            None,
            GObject.ParamFlags.READWRITE,
        ),
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
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # properties

        # default implementation of drawing.
        self._module: Optional[str] = None
        self._class_name: Optional[str] = None

        self._kwargs: Optional[str] = None
        self._location: Optional[str] = None

        # nvdsosd pads = bin pads by default (no location)
        self._artist = Gst.ElementFactory.make('pyfunc')
        self._artist.set_property('module', self._module)
        self._artist.set_property('class', self._class_name)
        self._artist.set_property('kwargs', self._kwargs)
        self.add(self._artist)
        drawer_sink_pad = self._artist.get_static_pad('sink')
        sink_pad = Gst.GhostPad.new_from_template(
            'sink', drawer_sink_pad, self.__gsttemplates__[0]
        )
        assert sink_pad.set_active(True)
        self.add_pad(sink_pad)

        drawer_src_pad = self._artist.get_static_pad('src')
        src_pad = Gst.GhostPad.new_from_template(
            'src', drawer_src_pad, self.__gsttemplates__[1]
        )
        assert src_pad.set_active(True)
        self.add_pad(src_pad)

    def do_get_property(self, prop: GObject.GParamSpec) -> Any:
        """Gst plugin get property function.

        :param prop: structure that encapsulates the parameter info
        """
        if prop.name == 'location':
            return self._location
        if prop.name == 'module':
            return self._module
        if prop.name == 'class':
            return self._class_name
        if prop.name == 'kwargs':
            return self._kwargs
        raise AttributeError(f'Unknown property {prop.name}.')

    def do_set_property(self, prop: GObject.GParamSpec, value: Any):
        """Gst plugin set property function.

        :param prop: structure that encapsulates the parameter info
        :param value: new value for parameter, type dependents on parameter
        """
        if prop.name == 'location':
            self._location = value
            self._parse_location()
        elif prop.name == 'module':
            self._module = value
            self._artist.set_property('module', value)
        elif prop.name == 'class':
            self._class_name = value
            self._artist.set_property('class', value)
        elif prop.name == 'kwargs':
            self._artist.set_property('kwargs', value)
            self._kwargs = value
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def _parse_location(self):
        if not self._location:
            return

        # TODO: check changes, if changed - destroy old, create new
        self.remove_pad(self.get_static_pad('src'))

        # tee.src_0 -> queue -> fakesink
        #   `.src_1 -> queue -> nvstreamdemux.src_0 -> queue -> nvvideoconvert -> ...

        tee = Gst.ElementFactory.make('tee')
        self.add(tee)
        self._artist.link(tee)

        # bin src pad branch
        bin_output_queue = Gst.ElementFactory.make('queue', 'tee_queue_0')
        self.add(bin_output_queue)
        tee_src_pad = tee.get_request_pad('src_%u')
        bin_output_queue_sink_pad = bin_output_queue.get_static_pad('sink')
        assert tee_src_pad.link(bin_output_queue_sink_pad) == Gst.PadLinkReturn.OK

        src_pad = Gst.GhostPad.new_from_template(
            'src', bin_output_queue.get_static_pad('src'), self.__gsttemplates__[1]
        )
        assert src_pad.set_active(True)
        self.add_pad(src_pad)

        # location processing branch
        tee_queue_1 = Gst.ElementFactory.make('queue', 'tee_queue_1')
        self.add(tee_queue_1)

        tee_src_pad = tee.get_request_pad('src_%u')
        tee_queue_1_sink_pad = tee_queue_1.get_static_pad('sink')
        assert tee_src_pad.link(tee_queue_1_sink_pad) == Gst.PadLinkReturn.OK

        # FIXME: exclude nvstreamdemux?
        # demux = Gst.ElementFactory.make('nvstreamdemux')
        # self.add(demux)
        # tee_queue_1.link(demux)
        #
        # # demuxer.src_0 only
        # demux_queue_0 = Gst.ElementFactory.make('queue', 'demux_queue')
        # self.add(demux_queue_0)
        #
        # demux_src_pad = demux.get_request_pad('src_0')
        # demux_queue_0_sink_pad = demux_queue_0.get_static_pad('sink')
        # assert demux_src_pad.link(demux_queue_0_sink_pad) == Gst.PadLinkReturn.OK

        nvvideoconvert = Gst.ElementFactory.make('nvvideoconvert')
        self.add(nvvideoconvert)
        # demux_queue_0.link(nvvideoconvert)
        tee_queue_1.link(nvvideoconvert)

        # output on display
        if self._location == 'display':
            sink = Gst.ElementFactory.make('nveglglessink')
            sink.set_property('enable-last-sample', 0)
            self.add(sink)
            sink.set_property('sync', 0)
            sink.set_property('qos', 0)
            if is_aarch64():
                transform = Gst.ElementFactory.make('nvegltransform')
                self.add(transform)
                nvvideoconvert.link(transform)
                transform.link(sink)
            else:
                nvvideoconvert.link(sink)

        # save output frames as jpeg images
        elif self._location.endswith('.jpg'):
            location_basedir = Path(self._location).parent
            shutil.rmtree(location_basedir, ignore_errors=True)
            location_basedir.mkdir(parents=True, exist_ok=True)

            jpegenc = Gst.ElementFactory.make('jpegenc')
            self.add(jpegenc)
            nvvideoconvert.link(jpegenc)

            sink = Gst.ElementFactory.make('multifilesink')
            sink.set_property('enable-last-sample', 0)
            self.add(sink)
            sink.set_property('location', self._location)
            jpegenc.link(sink)

        # save output as mp4/flv video
        elif self._location.endswith('.mp4') or self._location.endswith('.flv'):
            h264enc = Gst.ElementFactory.make('nvv4l2h264enc')
            self.add(h264enc)
            nvvideoconvert.link(h264enc)
            h264parse = Gst.ElementFactory.make('h264parse')
            self.add(h264parse)
            h264enc.link(h264parse)
            muxer = Gst.ElementFactory.make(
                'qtmux' if self._location.endswith('.mp4') else 'flvmux'
            )
            self.add(muxer)
            h264parse.link(muxer)
            sink = Gst.ElementFactory.make('filesink')
            sink.set_property('enable-last-sample', 0)
            self.add(sink)
            sink.set_property('location', self._location)
            muxer.link(sink)

        # unknown location
        else:
            sink = Gst.ElementFactory.make('fakesink')
            sink.set_property('enable-last-sample', 0)
            self.add(sink)
            nvvideoconvert.link(sink)


# register plugin
GObject.type_register(DrawBin)
__gstelementfactory__ = (
    DrawBin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    DrawBin,
)
