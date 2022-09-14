"""ZeroMQ properties."""
from savant.gstreamer import GObject


def socket_type_property(enum_type):
    """Gst property for ZeroMQ socket type factory."""
    return (
        # TODO: make enum
        GObject.TYPE_STRING,
        'ZeroMQ socket type',
        'ZeroMQ socket type (allowed: '
        f'{", ".join([enum_member.name for enum_member in enum_type])})',
        None,
        GObject.ParamFlags.READWRITE,
    )


ZEROMQ_PROPERTIES = {
    'socket': (
        GObject.TYPE_STRING,
        'ZeroMQ socket',
        'ZeroMQ socket',
        None,
        GObject.ParamFlags.READWRITE,
    ),
    'bind': (
        bool,
        'Bind socket',
        'Bind socket',
        True,
        GObject.ParamFlags.READWRITE,
    ),
}
