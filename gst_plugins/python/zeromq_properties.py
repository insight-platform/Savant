"""ZeroMQ properties."""

from savant.gstreamer import GObject


ZEROMQ_PROPERTIES = {
    'socket': (
        GObject.TYPE_STRING,
        'ZeroMQ socket',
        'ZeroMQ socket',
        None,
        GObject.ParamFlags.READWRITE,
    ),
}
