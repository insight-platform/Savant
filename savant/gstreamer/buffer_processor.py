"""Buffer processor for GStreamer pipeline."""
import inspect
from abc import ABC, abstractmethod
from queue import Queue
from types import FrameType
from typing import Iterator

from gi.repository import Gst

from savant.config.schema import PipelineElement
from savant.gstreamer.utils import gst_post_stream_failed_error
from savant.utils.fps_meter import FPSMeter
from savant.utils.logging import get_logger
from savant.utils.sink_factories import SinkMessage


class GstBufferProcessor(ABC):
    def __init__(self, queue: Queue, fps_meter: FPSMeter):
        """Buffer processor for GStreamer pipeline.

        :param queue: Queue for output data.
        :param fps_meter: FPS meter.
        """

        self._logger = get_logger(
            f'{self.__class__.__module__}.{self.__class__.__name__}'
        )
        self._queue = queue
        self._fps_meter = fps_meter

    # Buffer handlers
    @abstractmethod
    def prepare_input(self, buffer: Gst.Buffer):
        """Pipeline input processor."""

    @abstractmethod
    def prepare_output(self, buffer: Gst.Buffer, user_data) -> Iterator[SinkMessage]:
        """Pipeline output processor."""

    @abstractmethod
    def on_eos(self, user_data):
        """Pipeline EOS handler."""

    @abstractmethod
    def prepare_element_input(self, element: PipelineElement, buffer: Gst.Buffer):
        """Element input processor."""

    @abstractmethod
    def prepare_element_output(self, element: PipelineElement, buffer: Gst.Buffer):
        """Element output processor."""

    # Pad probes to attach buffer handlers
    def input_probe(  # pylint: disable=unused-argument
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
    ):
        """Attach pipeline input processor to pad."""

        buffer = info.get_buffer()
        try:
            self.prepare_input(buffer)
        except Exception as exc:  # pylint: disable=broad-except
            self._report_error(
                pad,
                f'Failed to prepare input for buffer with PTS {buffer.pts}: {exc}.',
                inspect.currentframe(),
            )

        return Gst.PadProbeReturn.OK

    def output_probe(  # pylint: disable=unused-argument
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        user_data,
    ):
        """Attach pipeline output processor to pad."""

        buffer = info.get_buffer()
        try:
            for sink_message in self.prepare_output(buffer, user_data):
                self._queue.put(sink_message)
                # measure and logging FPS
                if self._fps_meter():
                    self._logger.info(self._fps_meter.message)
        except Exception as exc:  # pylint: disable=broad-except
            self._report_error(
                pad,
                f'Failed to prepare output for buffer with PTS {buffer.pts}: {exc}.',
                inspect.currentframe(),
            )

        return Gst.PadProbeReturn.OK

    def element_input_probe(  # pylint: disable=unused-argument
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        element: PipelineElement,
    ):
        """Attach element input processor to pad."""

        buffer = info.get_buffer()
        try:
            self.prepare_element_input(element, buffer)
        except Exception as exc:  # pylint: disable=broad-except
            self._report_error(
                pad,
                f'Failed to prepare "{element.name}" element input for buffer with PTS {buffer.pts}: {exc}.',
                inspect.currentframe(),
            )

        return Gst.PadProbeReturn.OK

    def element_output_probe(  # pylint: disable=unused-argument
        self,
        pad: Gst.Pad,
        info: Gst.PadProbeInfo,
        element: PipelineElement,
    ):
        """Attach element output processor to pad."""

        buffer = info.get_buffer()
        try:
            self.prepare_element_output(element, buffer)
        except Exception as exc:  # pylint: disable=broad-except
            self._report_error(
                pad,
                f'Failed to prepare "{element.name}" element output for buffer with PTS {buffer.pts}: {exc}.',
                inspect.currentframe(),
            )

        return Gst.PadProbeReturn.OK

    def _report_error(self, pad: Gst.Pad, error: str, frame: FrameType):
        self._logger.exception(error)
        gst_post_stream_failed_error(
            gst_element=pad.get_parent_element(),
            frame=frame,
            file_path=__file__,
            text=error,
        )
