"""GStreamer utils."""
import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict

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


class LoggerMixin:
    """Mixes logger in GStreamer element.

    When the element name is available, logger name changes to
    `module_name/element_name`. Otherwise, logger name is `module_name`.

    Note: we cannot override `do_set_state` or any other method where element name
    becomes available since base classes are bindings.
    """

    _logger: logging.Logger = None
    _logger_initialized: bool = False

    def __init__(self):
        self._init_logger()

    @property
    def logger(self):
        """Logger."""
        if not self._logger_initialized:
            self._init_logger()
        return self._logger

    def _init_logger(self):
        if hasattr(self, 'get_name') and self.get_name():
            self._logger = logging.getLogger(
                f'savant.{self.__module__}.{self.get_name()}'
            )
        else:
            self._logger = logging.getLogger(f'savant.{self.__module__}')

        self._logger_initialized = True


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
