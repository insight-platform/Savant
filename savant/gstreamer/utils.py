"""GStreamer utils."""
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

from gi.repository import Gst  # noqa:F401
from savant.gstreamer.ffi import LIBGST, GstMapInfo


@contextmanager
def map_gst_buffer(
    gst_buffer: Gst.Buffer, flags: int = Gst.MapFlags.READ
) -> GstMapInfo:
    """Check if the buffer is writable and try to map it. Unmap at context
    exit.

    :param gst_buffer: Gst.Buffer object from GStreamer bindings
    :param flags: Gst.MapFlags for read/write map mode, READ by default
    :return: GstMapInfo structure
    """
    gst_buffer_p = hash(gst_buffer)
    map_info = GstMapInfo()

    if (
        flags & Gst.MapFlags.WRITE
        and LIBGST.gst_mini_object_is_writable(gst_buffer_p) == 0
    ):
        raise ValueError('Writable array requested but buffer is not writeable')

    success = LIBGST.gst_buffer_map(gst_buffer_p, map_info, flags)
    if not success:
        raise RuntimeError("Couldn't map buffer")
    try:
        yield map_info
    finally:
        LIBGST.gst_buffer_unmap(gst_buffer_p, map_info)


def pad_to_source_id(pad: Gst.Pad) -> str:
    """Extract source ID from pad name.

    Pad should be named with pattern "src_<source_id>" (eg "src_cam-1").
    """
    return pad.get_name()[4:]


def on_pad_event(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    event_handlers: Dict[
        Gst.EventType,
        Callable[[Gst.Pad, Gst.Event, Any], Gst.PadProbeReturn],
    ],
    *data,
):
    """Pad probe to handle events.

    Example::

        def on_caps(pad: Gst.Pad, event: Gst.Event, data):
            logger.info('Got caps %s from pad %s', event.parse_caps(), pad.get_name())
            return Gst.PadProbeReturn.OK

        pad.add_probe(
            Gst.PadProbeType.EVENT_DOWNSTREAM,
            on_pad_event,
            {Gst.EventType.EOS: on_eos},
            data
        )
    """

    if not info.type & Gst.PadProbeType.EVENT_BOTH:
        return Gst.PadProbeReturn.PASS
    event: Gst.Event = info.get_event()
    event_handler = event_handlers.get(event.type)
    if event_handler is None:
        return Gst.PadProbeReturn.PASS
    return event_handler(pad, event, *data)


def propagate_gst_setting_error(gst_element: Gst.Element, frame, file_path, text=None):
    propagate_gst_error(
        gst_element=gst_element,
        frame=frame,
        file_path=file_path,
        domain=Gst.LibraryError.quark(),
        code=Gst.LibraryError.SETTINGS,
        text=text,
    )


def propagate_gst_error(
    gst_element: Gst.Element,
    frame,
    file_path,
    domain,
    code,
    text=None,
    debug=None,
):
    gst_element.message_full(
        type=Gst.MessageType.ERROR,
        domain=domain,
        code=code,
        text=text,
        debug=debug,
        file=file_path,
        function=frame.f_code.co_name,
        line=frame.f_code.co_firstlineno,
    )


def gst_buffer_from_list(data: List[bytes]) -> Gst.Buffer:
    """Wrap list of data to GStreamer buffer."""

    buffer: Gst.Buffer = Gst.Buffer.new()
    for item in data:
        buffer = buffer.append(Gst.Buffer.new_wrapped(item))
    return buffer


class RequiredPropertyError(Exception):
    """Raised when required property is not set."""

    pass


def required_property(name: str, value: Optional[Any]):
    """Check if the property is set."""

    if value is None:
        raise RequiredPropertyError(f'"{name}" property is required')
