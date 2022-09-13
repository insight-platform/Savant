"""GStreamer base pipeline."""
from queue import Queue, Empty as EmptyException
from typing import Any, List, Dict, Generator, Optional, Tuple
import logging
from gi.repository import Gst  # noqa:F401

from savant.parameter_storage import param_storage
from savant.config.schema import PipelineElement, ModelElement
from savant.utils.sink_factories import SinkMessage
from savant.utils.fps_meter import FPSMeter


class CreateElementException(Exception):
    """Unable to create Gst.Element Exception."""


class GstPipeline:  # pylint: disable=too-many-instance-attributes
    """Base class for managing GStreamer based pipelines (DeepStream, DL
    Streamer).

    :param name: Pipeline name.
    :param source: Pipeline source element.
    :param elements: Pipeline elements.
    :key queue_maxsize: Output queue size.
    :key fps_period: FPS measurement period, in frames.
    """

    def __init__(
        self,
        name: str,
        source: PipelineElement,
        elements: List[PipelineElement],
        **kwargs,
    ):
        self._logger = logging.getLogger(f'savant.{name}')

        # output messages queue
        self._queue = Queue(maxsize=kwargs['queue_maxsize'])

        # init FPS meter
        self._fps_meter = FPSMeter(period_frames=kwargs['fps_period'])

        # init pipeline
        self._pipeline: Gst.Pipeline = Gst.Pipeline(name)

        # explicitly added elements container
        self._elements: List[Tuple[PipelineElement, Gst.Element]] = []

        # last added element - to link elements properly
        # `last_element.link(new_element)`
        self._last_element: Gst.Element = None

        # build pipeline: source -> elements -> fakesink
        self._add_source(source)
        for element in elements:
            self._add_element(element, with_probes=isinstance(element, ModelElement))
        self._add_sink()

        self._is_running = False

    def __str__(self) -> str:
        elements = ' -> '.join([e.full_name for e, _ in self.elements])
        if not elements:
            elements = 'no elements'
        return f'{self._pipeline.name}<{self.__class__.__name__}>: {elements}'

    @staticmethod
    def create_gst_element(element: PipelineElement) -> Gst.Element:
        """Creates Gst.Element.

        :param element: pipeline element to create.
        """
        gst_element = Gst.ElementFactory.make(element.element, element.name)
        if not gst_element:
            raise CreateElementException(f'Unable to create element {element}.')

        for prop_name, prop_value in element.properties.items():
            gst_element.set_property(prop_name, prop_value)

        for prop_name, dyn_gst_prop in element.dynamic_properties.items():

            def on_change(response_value: Any, propery_name: str = prop_name):
                gst_element.set_property(propery_name, response_value)

            param_storage().register_dynamic_parameter(
                dyn_gst_prop.storage_key, dyn_gst_prop.default, on_change
            )
            prop_value = param_storage()[dyn_gst_prop.storage_key]
            gst_element.set_property(prop_name, prop_value)

        return gst_element

    def _add_element(
        self,
        element: PipelineElement,
        with_probes: bool = False,
        link: bool = True,
    ) -> Gst.Element:
        """Creates, adds to pipeline and links element to the last one."""
        if element.name and self._pipeline.get_by_name(element.name):
            raise CreateElementException(
                f'Duplicate element name {element} in the pipeline.'
            )

        gst_element = self.create_gst_element(element)
        self._pipeline.add(gst_element)
        if link and self._last_element:
            self._last_element.link(gst_element)
        self._last_element = gst_element

        # set element name from GstElement
        if element.name is None:
            element.name = gst_element.name

        self._elements.append((element, gst_element))
        self._logger.debug('Added element %s: %s.', element.full_name, element)

        if with_probes:
            gst_element.get_static_pad('sink').add_probe(
                Gst.PadProbeType.BUFFER, self._element_input_probe, element
            )
            gst_element.get_static_pad('src').add_probe(
                Gst.PadProbeType.BUFFER, self._element_output_probe, element
            )
            self._logger.debug('Added in/out probes to element %s.', element.full_name)

        return gst_element

    def _element_input_probe(  # pylint: disable=unused-argument
        self, pad: Gst.Pad, info: Gst.PadProbeInfo, element: PipelineElement
    ):
        buffer = info.get_buffer()
        self.prepare_element_input(element, buffer)
        return Gst.PadProbeReturn.OK

    def prepare_element_input(self, element: PipelineElement, buffer: Gst.Buffer):
        """Element input processor."""

    def _element_output_probe(  # pylint: disable=unused-argument
        self, pad: Gst.Pad, info: Gst.PadProbeInfo, element: PipelineElement
    ):
        buffer = info.get_buffer()
        self.prepare_element_output(element, buffer)
        return Gst.PadProbeReturn.OK

    def prepare_element_output(self, element: PipelineElement, buffer: Gst.Buffer):
        """Element output processor."""

    def _add_source(self, source: PipelineElement) -> Gst.Element:
        source.name = 'source'
        _source = self._add_element(source)
        # input processor (post-source)
        _source.get_static_pad('src').add_probe(
            Gst.PadProbeType.BUFFER, self._input_probe, 0
        )
        return _source

    # pylint:disable=keyword-arg-before-vararg)
    def _add_sink(
        self,
        sink: Optional[PipelineElement] = None,
        link: bool = True,
        *probe_data,
    ) -> Gst.Element:
        if not sink:
            sink = PipelineElement(
                element='fakesink', name='sink', properties=dict(sync=0, qos=0)
            )

        _sink = self._add_element(sink, link=link)

        # output processor (pre-sink)
        _sink.get_static_pad('sink').add_probe(
            Gst.PadProbeType.BUFFER, self._output_probe, *probe_data
        )

        return _sink

    def _input_probe(  # pylint: disable=unused-argument
        self, pad: Gst.Pad, info: Gst.PadProbeInfo, user_data: Any
    ):
        buffer = info.get_buffer()
        self.prepare_input(buffer)
        return Gst.PadProbeReturn.OK

    def prepare_input(self, buffer: Gst.Buffer):
        """Pipeline input processor."""

    def _output_probe(  # pylint: disable=unused-argument
        self, pad: Gst.Pad, info: Gst.PadProbeInfo, *data
    ):
        buffer = info.get_buffer()
        self.prepare_output(buffer, *data)
        # measure and logging FPS
        if self._fps_meter():
            self._log_fps()
        return Gst.PadProbeReturn.OK

    def prepare_output(self, buffer: Gst.Buffer, *data):
        """Pipeline output processor."""

    def on_startup(self):
        """Callback called after pipeline is set to PLAYING."""
        self._is_running = True
        # start fps meter
        self._fps_meter.start()

    def on_shutdown(self):
        """Callback called after pipeline is set to NULL."""
        self._is_running = False
        self._log_fps()

    def _log_fps(self):
        self._logger.info(self._fps_meter.message)

    @property
    def elements(self) -> List[Tuple[PipelineElement, Gst.Element]]:
        """Pipeline elements.

        :return: Pipeline elements.
        """
        return self._elements

    def get_bus(self) -> Gst.Bus:
        """Get underlying Gst.Pipeline bus.

        :return: gstreamer bus of this pipeline.
        """
        return self._pipeline.get_bus()

    def set_state(self, state: Gst.State) -> None:
        """Set underlying Gst.Pipeline state.

        :param state: One of Gst element States (READY, PLAYING etc.)
            to set in underlying pipeline
        """
        self._pipeline.set_state(state)

    def stream(self, timeout: float = 0.1) -> Generator[SinkMessage, None, None]:
        """Fetches messages from queue using specified timeout (sec).

        :param timeout: queue get timeout.
        :return: output message.
        """
        while not self._queue.empty() or self._is_running:
            try:
                yield self._queue.get(timeout=timeout)
            except EmptyException:
                pass
