import time
from threading import Thread
from typing import Callable, List, Optional

from adapters.ds.sinks.always_on_rtsp.config import Config
from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrame
from savant.config.schema import PipelineElement
from savant.gstreamer import Gst
from savant.gstreamer.element_factory import GstElementFactory
from savant.gstreamer.runner import GstPipelineRunner
from savant.utils.logging import get_logger

LOGGER_NAME = 'adapters.ao_sink.pipeline'


def add_elements(
    pipeline: Gst.Pipeline,
    elements: List[PipelineElement],
    factory: GstElementFactory,
) -> List[Gst.Element]:
    gst_elements: List[Gst.Element] = []
    for element in elements:
        gst_element = factory.create(element)
        pipeline.add(gst_element)
        if gst_elements:
            assert gst_elements[-1].link(gst_element)
        gst_elements.append(gst_element)
    return gst_elements


class PipelineThread:
    def __init__(
        self,
        build_pipeline: Callable[[Config, LastFrame, GstElementFactory], Gst.Pipeline],
        thread_name: str,
        config: Config,
        last_frame: LastFrame,
        factory: GstElementFactory,
    ):
        self.build_pipeline = build_pipeline
        self.thread_name = thread_name
        self.config = config
        self.last_frame = last_frame
        self.factory = factory

        self.is_running = False
        self.thread: Optional[Thread] = None
        self.logger = get_logger(f'{LOGGER_NAME}.{self.__class__.__name__}')

    def start(self):
        self.is_running = True
        self.thread = Thread(name=self.thread_name, target=self.workload)
        self.thread.start()

    def stop(self):
        self.is_running = False

    def join(self):
        self.thread.join()

    def workload(self):
        pipeline = self.build_pipeline(self.config, self.last_frame, self.factory)
        self.logger.info('Starting pipeline %s', pipeline.get_name())
        with GstPipelineRunner(pipeline) as runner:
            while self.is_running and runner._is_running:
                time.sleep(1)
        self.logger.info('Pipeline %s is stopped', pipeline.get_name())
        self.is_running = False
