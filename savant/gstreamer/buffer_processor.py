"""Buffer processor for GStreamer pipeline."""
from abc import ABC, abstractmethod
from queue import Queue
from typing import Iterator

from gi.repository import Gst

from savant.utils.logging import get_logger
from savant.utils.sink_factories import SinkMessage, SinkVideoFrame


class GstBufferProcessor(ABC):
    def __init__(self, queue: Queue):
        """Buffer processor for GStreamer pipeline.

        :param queue: Queue for output data.
        """

        self._logger = get_logger(
            f'{self.__class__.__module__}.{self.__class__.__name__}'
        )
        self._queue = queue

    @property
    def logger(self):
        return self._logger

    # Buffer handlers
    @abstractmethod
    def prepare_input(self, buffer: Gst.Buffer):
        """Pipeline input processor."""

    @abstractmethod
    def prepare_output(self, buffer: Gst.Buffer, user_data) -> Iterator[SinkMessage]:
        """Pipeline output processor."""

    def process_output(self, buffer: Gst.Buffer, user_data):
        """Pipeline output processor wrapper."""
        for sink_message in self.prepare_output(buffer, user_data):
            self._queue.put(sink_message)

    @abstractmethod
    def on_eos(self, user_data):
        """Pipeline EOS handler."""
