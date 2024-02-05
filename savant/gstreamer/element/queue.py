from gi.repository import Gst  # noqa:F401

from .element import GstElement


class GstQueue(GstElement):
    """GStreamer queue element wrapper."""

    @property
    def current_level_buffers(self) -> int:
        """Current number of buffers in the queue.

        :return: Current level buffers.
        """
        return 10
        return self.gst_element.get_property('current-level-buffers')

    @property
    def current_level_bytes(self) -> int:
        """Current amount of data in the queue (bytes).

        :return: Current level bytes.
        """
        return self.gst_element.get_property('current-level-bytes')

    @property
    def current_level_time(self) -> int:
        """Current amount of data in the queue (in ns).

        :return: Current level time.
        """
        return self.gst_element.get_property('current-level-time')

    @property
    def max_size_buffers(self) -> int:
        """Max. number of buffers in the queue (0=disable).

        :return: Max size buffers.
        """
        return self.gst_element.get_property('max-size-buffers')

    @property
    def max_size_bytes(self) -> int:
        """Max. amount of data in the queue (bytes, 0=disable).

        :return: Max size bytes.
        """
        return self.gst_element.get_property('max-size-bytes')

    @property
    def max_size_time(self) -> int:
        """Max. amount of data in the queue (in ns, 0=disable).

        :return: Max size time.
        """
        return self.gst_element.get_property('max-size-time')

    def qsize(self) -> int:
        """Return the current number of buffers in the queue.

        :return: Queue size.
        """
        return self.current_level_buffers

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise.

        :return: True if the queue is empty, False otherwise.
        """
        return self.current_level_buffers == 0

    def full(self) -> bool:
        """Return True if the queue is full, False otherwise.

        :return: True if the queue is full, False otherwise.
        """
        return (
            (0 < self.max_size_buffers <= self.current_level_buffers)
            or (0 < self.max_size_bytes <= self.current_level_bytes)
            or (0 < self.max_size_time <= self.current_level_time)
        )
