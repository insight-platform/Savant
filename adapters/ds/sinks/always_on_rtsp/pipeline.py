import time
from typing import Callable, List

from adapters.ds.sinks.always_on_rtsp.config import Config
from adapters.ds.sinks.always_on_rtsp.last_frame import LastFrameRef
from adapters.shared.thread import BaseThreadWorker
from savant.config.schema import PipelineElement
from savant.gstreamer import Gst
from savant.gstreamer.element_factory import GstElementFactory
from savant.gstreamer.runner import GstPipelineRunner

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


class PipelineThread(BaseThreadWorker):
    def __init__(
        self,
        build_pipeline: Callable[
            [Config, LastFrameRef, GstElementFactory], Gst.Pipeline
        ],
        thread_name: str,
        config: Config,
        last_frame: LastFrameRef,
        factory: GstElementFactory,
    ):
        super().__init__(
            thread_name=thread_name,
            logger_name=f'{LOGGER_NAME}.{self.__class__.__name__}',
        )
        self.build_pipeline = build_pipeline
        self.config = config
        self.last_frame = last_frame
        self.factory = factory

    def workload(self):
        pipeline = self.build_pipeline(self.config, self.last_frame, self.factory)
        self.logger.info(
            'Source %s. Starting pipeline %s',
            self.config.source_id,
            pipeline.get_name(),
        )
        with GstPipelineRunner(pipeline) as runner:
            while self.is_running and runner._is_running:
                time.sleep(1)
        self.logger.info(
            'Source %s. Pipeline %s is stopped',
            self.config.source_id,
            pipeline.get_name(),
        )
        self.is_running = False
