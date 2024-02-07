"""GStreamer utils."""
import inspect
from contextlib import contextmanager
from types import FrameType
from typing import Any, Callable, Dict, List, Optional

from gi.repository import Gst  # noqa:F401
from savant_rs.utils import ByteBuffer
from savant_rs.utils.serialization import Message, load_message_from_bytebuffer

from savant.gstreamer.ffi import LIBGST, GstMapInfo
from savant.utils.logging import get_logger

logger = get_logger(__name__)


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


def gst_post_library_settings_error(
    gst_element: Gst.Element,
    frame: Optional[FrameType],
    file_path: str,
    text: Optional[str] = None,
    debug: Optional[str] = None,
):
    """Post a library settings error message on the bus from inside an element."""
    gst_post_error(
        gst_element=gst_element,
        frame=frame,
        file_path=file_path,
        domain=Gst.LibraryError.quark(),
        code=Gst.LibraryError.SETTINGS,
        text=text,
        debug=debug,
    )


def gst_post_error(
    gst_element: Gst.Element,
    frame: Optional[FrameType],
    file_path: str,
    domain: int,
    code: int,
    text: Optional[str] = None,
    debug: Optional[str] = None,
):
    """Post an error message on the bus from inside an element."""
    gst_post_message(
        msg_type=Gst.MessageType.ERROR,
        gst_element=gst_element,
        frame=frame,
        file_path=file_path,
        domain=domain,
        code=code,
        text=text,
        debug=debug,
    )


def gst_post_warning(
    gst_element: Gst.Element,
    frame: Optional[FrameType],
    file_path: str,
    domain: int,
    code: int,
    text: Optional[str] = None,
    debug: Optional[str] = None,
):
    """Post a warning message on the bus from inside an element."""
    gst_post_message(
        msg_type=Gst.MessageType.WARNING,
        gst_element=gst_element,
        frame=frame,
        file_path=file_path,
        domain=domain,
        code=code,
        text=text,
        debug=debug,
    )


def gst_post_stream_failed_warning(
    gst_element: Gst.Element,
    frame: Optional[FrameType],
    file_path: str,
    text: Optional[str] = None,
    debug: Optional[str] = None,
):
    """Post a stream failed warning message on the bus from inside an element."""
    gst_post_warning(
        gst_element=gst_element,
        frame=frame,
        file_path=file_path,
        domain=Gst.StreamError.quark(),
        code=Gst.StreamError.FAILED,
        text=text,
        debug=debug,
    )


def gst_post_stream_failed_error(
    gst_element: Gst.Element,
    frame: Optional[FrameType],
    file_path: str,
    text: Optional[str] = None,
    debug: Optional[str] = None,
):
    """Post a stream failed error message on the bus from inside an element."""
    gst_post_error(
        gst_element=gst_element,
        frame=frame,
        file_path=file_path,
        domain=Gst.StreamError.quark(),
        code=Gst.StreamError.FAILED,
        text=text,
        debug=debug,
    )


def gst_post_stream_demux_error(
    gst_element: Gst.Element,
    frame: Optional[FrameType],
    file_path: str,
    text: Optional[str] = None,
    debug: Optional[str] = None,
):
    """Post a stream demux error message on the bus from inside an element."""
    gst_post_error(
        gst_element=gst_element,
        frame=frame,
        file_path=file_path,
        domain=Gst.StreamError.quark(),
        code=Gst.StreamError.DEMUX,
        text=text,
        debug=debug,
    )


def gst_post_message(
    msg_type: int,
    gst_element: Gst.Element,
    frame: Optional[FrameType],
    file_path: str,
    domain: int,
    code: int,
    text: Optional[str] = None,
    debug: Optional[str] = None,
):
    """Post an error, warning or info message on the bus from inside an element.

    :param msg_type: Must be one of Gst.MessageType.ERROR, Gst.MessageType.WARNING, Gst.MessageType.INFO.
    :param gst_element: Gst Element that posts the message.
    :param frame: Frame object from inspect.currentframe().
    :param file_path: Path to the file that posts the message.
    :param domain: The GStreamer error domain this error belongs to.
    :param code: The error code belonging to the domain, check here
        https://gstreamer.freedesktop.org/documentation/gstreamer/gsterror.html?gi-language=python
    :param text: Error text.
    :param debug: Debug info.
    """
    if frame:
        function = frame.f_code.co_name
        line = frame.f_code.co_firstlineno
    else:
        function = 'Unknown'
        line = 0
    gst_element.message_full(
        type=msg_type,
        domain=domain,
        code=code,
        text=text,
        debug=debug,
        file=file_path,
        function=function,
        line=line,
    )


def gst_buffer_from_list(data: List[bytes]) -> Gst.Buffer:
    """Wrap list of data to GStreamer buffer."""

    buffer: Gst.Buffer = Gst.Buffer.new()
    for item in data:
        buffer = buffer.append(Gst.Buffer.new_wrapped(item))
    return buffer


class RequiredPropertyError(Exception):
    """Raised when required property is not set."""


def required_property(name: str, value: Optional[Any]):
    """Check if the property is set."""

    if value is None:
        raise RequiredPropertyError(f'"{name}" property is required')


def link_pads(src_pad: Gst.Pad, sink_pad: Gst.Pad):
    """Link pads and raise exception if linking failed."""

    assert src_pad.link(sink_pad) == Gst.PadLinkReturn.OK, (
        f'Unable to link {src_pad.get_parent_element().get_name()}.{src_pad.get_name()} '
        f'to {sink_pad.get_parent_element().get_name()}.{sink_pad.get_name()}'
    )


def load_message_from_gst_buffer(buffer: Gst.Buffer) -> Message:
    frame_meta_mapinfo: Gst.MapInfo
    result, frame_meta_mapinfo = buffer.map_range(0, 1, Gst.MapFlags.READ)
    assert result, f'Cannot read buffer with PTS={buffer.pts}.'
    try:
        return load_message_from_bytebuffer(ByteBuffer(frame_meta_mapinfo.data))
    finally:
        buffer.unmap(frame_meta_mapinfo)


def add_buffer_probe(pad: Gst.Pad, callback: Callable, *data: Any):
    """Adds buffer probe with specified callback and args to the element pad.

    :param pad: Element pad (src/sink) to add probe to.
    :param callback: User callback to call in the probe.
    :param data: User data to pass to the user callback as args after the Gst.Buffer.
    """
    pad.add_probe(
        Gst.PadProbeType.BUFFER,
        _buffer_probe_callback,
        callback,
        *data,
    )


def _buffer_probe_callback(
    pad: Gst.Pad,
    info: Gst.PadProbeInfo,
    callback: Callable,
    *data: Any,
):
    """Buffer probe callback wrapper to handle exceptions."""
    buffer = info.get_buffer()
    try:
        callback(buffer, *data)
    except Exception as exc:  # pylint: disable=broad-except
        error = f'Failed to call {callback} for buffer with PTS {buffer.pts}: {exc}.'
        logger.exception(error)
        gst_post_stream_failed_error(
            gst_element=pad.get_parent_element(),
            frame=inspect.currentframe(),
            file_path=__file__,
            text=error,
        )

    return Gst.PadProbeReturn.OK


def get_elements(
    gst_bin: Gst.Bin,
    factory: Optional[str] = None,
    before_element: Optional[Gst.Element] = None,
    after_element: Optional[Gst.Element] = None,
) -> List[Gst.Element]:
    """Get Bin/Pipeline elements in order (source -> elem1 -> elem2 -> ... -> sink).

    :param gst_bin: Gst.Bin or Gst.Pipeline object.
    :param factory: Optional factory name to filter elements.
    :param before_element: Optional element to get elements before.
    :param after_element: Optional element to get elements after.
    :return: List of elements in order.
    """
    elements = []
    element_iter = gst_bin.iterate_sorted()
    while True:
        ret, element = element_iter.next()
        if ret != Gst.IteratorResult(1):  # GST_ITERATOR_OK
            break
        elements.append(element)
    elements.reverse()
    try:
        if before_element:
            elements = elements[: elements.index(before_element)]
        if after_element:
            elements = elements[elements.index(after_element) + 1 :]
    except ValueError:
        pass
    if factory:
        elements = [
            elem for elem in elements if elem.get_factory().get_name() == factory
        ]
    return elements
