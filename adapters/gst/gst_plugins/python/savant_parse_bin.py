"""SavantParseBin element."""
from typing import Optional

from savant.gstreamer import GObject, Gst  # noqa:F401
from savant.gstreamer.codecs import CODEC_BY_CAPS_NAME, Codec, CodecInfo
from savant.gstreamer.utils import on_pad_event
from savant.utils.logging import LoggerMixin

SINK_PAD_TEMPLATE = Gst.PadTemplate.new(
    'sink',
    Gst.PadDirection.SINK,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.from_string(';'.join(x.value.caps_name for x in Codec)),
)
SRC_PAD_TEMPLATE = Gst.PadTemplate.new(
    'src',
    Gst.PadDirection.SRC,
    Gst.PadPresence.ALWAYS,
    Gst.Caps.from_string(';'.join(x.value.caps_with_params for x in Codec)),
)


class SavantParseBin(LoggerMixin, Gst.Bin):
    """Applies an appropriate parser to the stream."""

    GST_PLUGIN_NAME = 'savant_parse_bin'

    __gstmetadata__ = (
        'Savant parse bin',
        'Generic/Bin/Parser',
        'Applies an appropriate parser to the stream.',
        'Pavel Tomskikh <tomskih_pa@bw-sw.com>',
    )

    __gsttemplates__ = (SINK_PAD_TEMPLATE, SRC_PAD_TEMPLATE)

    __gproperties__ = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parser: Optional[Gst.Element] = None
        self._sink_pad: Gst.Pad = Gst.GhostPad.new_no_target_from_template(
            'sink', SINK_PAD_TEMPLATE
        )
        self._src_pad: Gst.Pad = Gst.GhostPad.new_no_target_from_template(
            'src', SRC_PAD_TEMPLATE
        )
        self.add_pad(self._sink_pad)
        self.add_pad(self._src_pad)
        self._sink_pad.set_active(True)
        self._src_pad.set_active(True)

        self._sink_pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.CAPS: self.on_sink_pad_caps},
        )

    def on_sink_pad_caps(self, pad: Gst.Pad, event: Gst.Event):
        caps: Gst.Caps = event.parse_caps()
        self.logger.debug('Got caps %r. Trying to find parser.', caps.to_string())
        codec: CodecInfo = CODEC_BY_CAPS_NAME[caps[0].get_name()].value
        parser_name = codec.parser or 'identity'
        if parser_name == 'jpegparse':
            # JPEG parsing is not required in the adapter because the FFmpeg input element
            # sends full JPEG frames, not a continuous stream which requires parsing
            # Jpegparse also does not parse well for 4K yuvj420p probably because of the limitations.
            #
            self.logger.debug('JPEG parsing is not required in the adapter. Using identity.')
            parser_name = 'identity'
        self.logger.debug('Adding parser %r.', parser_name)
        self._parser = Gst.ElementFactory.make(parser_name)
        if parser_name in ['h264parse', 'h265parse']:
            # Send VPS, SPS and PPS with every IDR frame
            # h26xparse cannot start stream without VPS, SPS or PPS in the first frame
            self.logger.debug(
                'Set config-interval of %s to %s', self._parser.get_name(), -1
            )
            self._parser.set_property('config-interval', -1)
        self.add(self._parser)
        self._src_pad.set_target(self._parser.get_static_pad('src'))
        self._sink_pad.set_target(self._parser.get_static_pad('sink'))
        self._parser.sync_state_with_parent()
        self.logger.debug('Added parser %r.', self._parser.get_name())

        return Gst.PadProbeReturn.REMOVE


# register plugin
GObject.type_register(SavantParseBin)
__gstelementfactory__ = (
    SavantParseBin.GST_PLUGIN_NAME,
    Gst.Rank.NONE,
    SavantParseBin,
)
