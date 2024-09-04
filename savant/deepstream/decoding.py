from savant.gstreamer import GLib, GObject, Gst  # noqa:F401
from savant.utils.logging import get_logger
from savant.utils.platform import is_aarch64

logger = get_logger(__name__)


def configure_low_latency_decoding(element: Gst.Element):
    """Configure low latency decoding for the given element if element supports it."""

    factory_name = element.get_factory().get_name()
    if factory_name == 'nvv4l2decoder':
        logger.debug('Configuring low latency decoding for %s', element.get_name())
        if is_aarch64():
            element.set_property('enable-max-performance', True)
            element.set_property('disable-dpb', True)
        else:
            element.set_property('low-latency-mode', True)
