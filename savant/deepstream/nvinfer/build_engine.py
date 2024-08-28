import dataclasses
import time
from pathlib import Path

from savant.config.schema import ModelElement, PipelineElement
from savant.deepstream.nvinfer.model import NvInferModel
from savant.deepstream.runner import NvDsPipelineRunner
from savant.gstreamer import Gst  # noqa:F401
from savant.gstreamer.element_factory import CreateElementException, GstElementFactory


class BuildEngineException(Exception):
    """Unable to build engine Exception."""


def build_engine(element: ModelElement, rebuild: bool = True):
    """Builds the specified model engine (TRT).

    :param element: The model element for which the engine should be generated.
    :param rebuild: Flag to force regeneration if the engine already exists.
    :raises:
        CreateElementException: Failed to create or link elements.
        BuildEngineException: Failed to run the pipeline.
    """

    model: NvInferModel = element.model

    if rebuild:
        engine_file_path = Path(model.local_path) / model.engine_file
        engine_file_path.unlink(missing_ok=True)

    nvinfer_element = dataclasses.replace(element)
    nvinfer_element.properties['process-mode'] = 1  # primary

    pipeline: Gst.Pipeline = Gst.Pipeline()
    elements = [
        PipelineElement(
            'videotestsrc',
            properties={'num-buffers': model.batch_size},
        ),
        PipelineElement('nvvideoconvert'),
        PipelineElement(
            element='nvstreammux',
            name='muxer',
            properties={
                'width': model.input.width if model.input.width else 1280,
                'height': model.input.height if model.input.height else 720,
                'batch-size': 1,
            },
        ),
        nvinfer_element,
        PipelineElement('fakesink'),
    ]

    last_gst_element = None
    for element in elements:
        gst_element = GstElementFactory.create_element(element)
        pipeline.add(gst_element)
        if last_gst_element is not None:
            if element.element == 'nvstreammux':
                sink_pad = gst_element.request_pad_simple('sink_0')
                src_pad = last_gst_element.get_static_pad('src')
                link_res = src_pad.link(sink_pad) == Gst.PadLinkReturn.OK
            else:
                link_res = last_gst_element.link(gst_element)
            if not link_res:
                raise CreateElementException(
                    f'Failed to link {last_gst_element.name} to {gst_element.name}.'
                )
        last_gst_element = gst_element

    with NvDsPipelineRunner(pipeline) as runner:
        while runner.is_running:
            time.sleep(0.1)

    if runner.error is not None:
        raise BuildEngineException(runner.error)
