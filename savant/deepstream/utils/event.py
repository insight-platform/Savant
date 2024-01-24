"""DeepStream Custom Events API & Message Functions API Bindings
https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/group__gstreamer__nvevent.html

* gst-nvevent.h, libnvdsgst_helper.so
  defines GstNvEventType and create/parse functions

* gst-nvcustomevent.h, libnvdsgst_customhelper.so
  defines GstNvCustomEventType and create/parse functions
"""
import ctypes

from savant.gstreamer import GObject, Gst  # noqa:F401


def _make_nvevent_type(event_type: int):
    """Helper function to define custom event type (like GST_EVENT_MAKE_TYPE)."""
    return (event_type << Gst.EVENT_NUM_SHIFT) | (
        Gst.EventTypeFlags.DOWNSTREAM
        | Gst.EventTypeFlags.SERIALIZED
        | Gst.EventTypeFlags.STICKY
        | Gst.EventTypeFlags.STICKY_MULTI
    )


GST_NVEVENT_PAD_ADDED = _make_nvevent_type(400)
GST_NVEVENT_PAD_DELETED = _make_nvevent_type(401)
GST_NVEVENT_STREAM_EOS = _make_nvevent_type(402)
GST_NVEVENT_STREAM_SEGMENT = _make_nvevent_type(403)
GST_NVEVENT_STREAM_RESET = _make_nvevent_type(404)
GST_NVEVENT_STREAM_START = _make_nvevent_type(405)

_nv_event_type_name = {
    GST_NVEVENT_PAD_ADDED: 'GST_NVEVENT_PAD_ADDED',
    GST_NVEVENT_PAD_DELETED: 'GST_NVEVENT_PAD_DELETED',
    GST_NVEVENT_STREAM_EOS: 'GST_NVEVENT_STREAM_EOS',
    GST_NVEVENT_STREAM_SEGMENT: 'GST_NVEVENT_STREAM_SEGMENT',
    GST_NVEVENT_STREAM_RESET: 'GST_NVEVENT_STREAM_RESET',
    GST_NVEVENT_STREAM_START: 'GST_NVEVENT_STREAM_START',
}

GST_NVEVENT_ROI_UPDATE = _make_nvevent_type(406)
GST_NVEVENT_INFER_INTERVAL_UPDATE = _make_nvevent_type(407)

_nv_custom_event_type_name = {
    GST_NVEVENT_ROI_UPDATE: 'GST_NVEVENT_ROI_UPDATE',
    GST_NVEVENT_INFER_INTERVAL_UPDATE: 'GST_NVEVENT_INFER_INTERVAL_UPDATE',
}


def gst_event_type_to_str(event_type: int) -> str:
    """Returns event type string representation."""
    if event_type in _nv_event_type_name:
        return f'<enum {_nv_event_type_name[event_type]} of type GstNvEventType>'
    if event_type in _nv_custom_event_type_name:
        return (
            f'<enum {_nv_custom_event_type_name[event_type]} '
            'of type GstNvCustomEventType>'
        )
    return str(event_type)


def gst_nvevent_parse_pad_added(event: Gst.Event) -> int:
    """Extracts source-id (pad index) from GST_NVEVENT_PAD_ADDED event."""
    from .nvdsgst_helper import libnvdsgst_helper

    source_id = ctypes.c_uint()
    libnvdsgst_helper.gst_nvevent_parse_pad_added(hash(event), ctypes.byref(source_id))
    return source_id.value


def gst_nvevent_parse_pad_deleted(event: Gst.Event) -> int:
    """Extracts source-id (pad index) from GST_NVEVENT_PAD_DELETED event."""
    from .nvdsgst_helper import libnvdsgst_helper

    source_id = ctypes.c_uint()
    libnvdsgst_helper.gst_nvevent_parse_pad_deleted(
        hash(event), ctypes.byref(source_id)
    )
    return source_id.value


def gst_nvevent_parse_stream_eos(event: Gst.Event) -> int:
    """Extracts source-id (pad index) from GST_NVEVENT_STREAM_EOS event."""
    from .nvdsgst_helper import libnvdsgst_helper

    source_id = ctypes.c_uint()
    libnvdsgst_helper.gst_nvevent_parse_stream_eos(hash(event), ctypes.byref(source_id))
    return source_id.value


def gst_nvevent_parse_stream_start(event: Gst.Event) -> int:
    """Extracts source-id (pad index) from GST_NVEVENT_STREAM_START event."""
    from .nvdsgst_helper import libnvdsgst_helper

    source_id = ctypes.c_uint()
    libnvdsgst_helper.gst_nvevent_parse_stream_start(
        hash(event), ctypes.byref(source_id)
    )
    return source_id.value


def gst_nvevent_new_stream_eos(source_id: int) -> Gst.Event:
    """Creates a "custom EOS" event for the specified source.

    :param source_id: Source ID of the stream for which EOS is to be sent;
                      also the pad ID  of the sinkpad of the
                      Gst-nvstreammux plugin for which
                      the source is configured.
    """

    struct: Gst.Structure = Gst.Structure.new_empty('nv-stream-eos')
    struct.set_value('source-id', GObject.Value(GObject.TYPE_UINT, source_id))
    event: Gst.Event = Gst.Event.new_custom(Gst.EventType.UNKNOWN, struct)
    event.type = GST_NVEVENT_STREAM_EOS
    return event
