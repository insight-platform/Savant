from gi.repository import Gst  # noqa:F401


class GstElement:
    """GStreamer element wrapper."""

    def __init__(self, gst_element: Gst.Element):
        """Initializes GStreamer element wrapper.

        :param gst_element: GStreamer element.
        """
        self.gst_element = gst_element

    @property
    def name(self) -> str:
        """Get element name.

        :return: Element name.
        """
        return self.gst_element.get_name()
