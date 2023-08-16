"""GStreamer base pipeline."""
from queue import Queue, Empty as EmptyException
from typing import Any, List, Generator, Optional, Tuple
import logging
from gi.repository import Gst  # noqa:F401

from savant.gstreamer.buffer_processor import GstBufferProcessor
from savant.config.schema import (
    PipelineElement,
    Pipeline,
    ModelElement,
    ElementGroup,
)
from savant.utils.sink_factories import SinkMessage
from savant.utils.fps_meter import FPSMeter
from savant.gstreamer.element_factory import CreateElementException, GstElementFactory
from savant.utils.logging import get_logger


class GstPipeline:  # pylint: disable=too-many-instance-attributes
    """Base class for managing GStreamer based pipelines (DeepStream, DL
    Streamer).

    :param name: Pipeline name.
    :param pipeline_cfg: Pipeline config.
    :key queue_maxsize: Output queue size.
    :key fps_period: FPS measurement period, in frames.
    """

    # pipeline element factory
    _element_factory = GstElementFactory()

    def __init__(
        self,
        name: str,
        pipeline_cfg: Pipeline,
        **kwargs,
    ):
        self._logger = get_logger(f'savant.{name}')

        # output messages queue
        self._queue = Queue(maxsize=kwargs['queue_maxsize'])

        # init FPS meter
        self._fps_meter = FPSMeter(period_frames=kwargs['fps_period'])

        # create buffer processor
        self._buffer_processor = self._build_buffer_processor(
            self._queue, self._fps_meter
        )

        # init pipeline
        self._pipeline: Gst.Pipeline = Gst.Pipeline(name)

        # explicitly added elements container
        self._elements: List[Tuple[PipelineElement, Gst.Element]] = []

        # last added element - to link elements properly
        # `last_element.link(new_element)`
        self._last_element: Gst.Element = None

        # build pipeline: source -> elements -> fakesink
        self._logger.debug('Adding source...')
        self._add_source(pipeline_cfg.source)

        self._logger.debug('Adding pipeline elements...')
        for i, item in enumerate(pipeline_cfg.elements):
            if isinstance(item, PipelineElement):
                self.add_element(item, with_probes=isinstance(item, ModelElement))
            elif isinstance(item, ElementGroup):
                if self._is_group_enabled_check_log(item, i):
                    for element in item.elements:
                        self.add_element(
                            element, with_probes=isinstance(element, ModelElement)
                        )

        self._logger.debug('Adding sink...')
        self._add_sink()

        self._is_running = False

    def __str__(self) -> str:
        elements = ' -> '.join([e.full_name for e, _ in self.elements])
        if not elements:
            elements = 'no elements'
        return f'{self._pipeline.name}<{self.__class__.__name__}>: {elements}'

    def add_element(
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

        gst_element = self._element_factory.create(element)
        self._pipeline.add(gst_element)
        if link and self._last_element:
            assert self._last_element.link(
                gst_element
            ), f'Unable to link {element.name} to {self._last_element.name}'
        self._last_element = gst_element

        # set element name from GstElement
        if element.name is None:
            element.name = gst_element.name

        self._elements.append((element, gst_element))
        self._logger.debug('Added element %s: %s.', element.full_name, element)

        if with_probes:
            gst_element.get_static_pad('sink').add_probe(
                Gst.PadProbeType.BUFFER,
                self._buffer_processor.element_input_probe,
                element,
            )
            gst_element.get_static_pad('src').add_probe(
                Gst.PadProbeType.BUFFER,
                self._buffer_processor.element_output_probe,
                element,
            )
            self._logger.debug('Added in/out probes to element %s.', element.full_name)

        return gst_element

    def _add_source(self, source: PipelineElement) -> Gst.Element:
        source.name = 'source'
        _source = self.add_element(source)
        # input processor (post-source)
        _source.get_static_pad('src').add_probe(
            Gst.PadProbeType.BUFFER, self._buffer_processor.input_probe
        )
        return _source

    def _add_sink(
        self,
        sink: Optional[PipelineElement] = None,
        link: bool = True,
        probe_data: Any = None,
    ) -> Gst.Element:
        if not sink:
            sink = PipelineElement(
                element='fakesink', name='sink', properties=dict(sync=0, qos=0)
            )

        _sink = self.add_element(sink, link=link)

        # output processor (pre-sink)
        _sink.get_static_pad('sink').add_probe(
            Gst.PadProbeType.BUFFER, self._buffer_processor.output_probe, probe_data
        )

        return _sink

    def on_startup(self):
        """Callback called after pipeline is set to PLAYING."""
        self._is_running = True
        # start fps meter
        self._fps_meter.start()

    def before_shutdown(self):
        """Callback called before pipeline is set to NULL."""
        self._is_running = False

    def on_shutdown(self):
        """Callback called after pipeline is set to NULL."""
        self._log_fps()

    def _log_fps(self):
        self._logger.info(self._fps_meter.message)

    @property
    def elements(self) -> List[Tuple[PipelineElement, Gst.Element]]:
        """Pipeline elements.

        :return: Pipeline elements.
        """
        return self._elements

    @property
    def pipeline(self) -> Gst.Pipeline:
        """Get underlying Gst.Pipeline instance.

        :return: Gst.Pipeline instance of this pipeline.
        """
        return self._pipeline

    def get_bus(self) -> Gst.Bus:
        """Get underlying Gst.Pipeline bus.

        :return: gstreamer bus of this pipeline.
        """
        return self._pipeline.get_bus()

    def set_state(self, state: Gst.State) -> Gst.StateChangeReturn:
        """Set underlying Gst.Pipeline state.

        :param state: One of Gst element States (READY, PLAYING etc.)
            to set in underlying pipeline
        :return: Result of the state change using Gst.StateChangeReturn
        """
        return self._pipeline.set_state(state)

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

    def _build_buffer_processor(
        self,
        queue: Queue,
        fps_meter: FPSMeter,
    ) -> GstBufferProcessor:
        """Create buffer processor."""
        return GstBufferProcessor(queue, fps_meter)

    def _is_group_enabled_check_log(self, group: ElementGroup, group_idx: int) -> bool:
        is_enabled = group.init_condition.is_enabled
        if self._logger.isEnabledFor(logging.DEBUG):
            if is_enabled:
                cmp_symbol = '=='
                action = 'adding'

            else:
                cmp_symbol = '!='
                action = 'skipping'
            debug_str = f'expr "{group.init_condition.expr}" {cmp_symbol} value "{group.init_condition.value}", {action}'
            self._logger.debug(
                'Group %s (name %s): %s %s elements.',
                group_idx,
                group.name,
                debug_str,
                len(group.elements),
            )
        return is_enabled
