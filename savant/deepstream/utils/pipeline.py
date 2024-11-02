from typing import Any, Dict, List, Union

from savant_rs.pipeline2 import StageFunction, VideoPipelineStagePayloadType

from savant.config.schema import (
    BufferQueuesParameters,
    ElementGroup,
    Pipeline,
    PipelineElement,
    PyFuncElement,
)
from savant.utils.logging import get_logger

logger = get_logger(__name__)


def add_queues_to_pipeline(
    pipeline_cfg: Pipeline,
    buffer_queues: BufferQueuesParameters,
):
    """Add queues to the pipeline before and after pyfunc elements."""

    queue_properties = {
        'max-size-buffers': buffer_queues.length,
        'max-size-bytes': buffer_queues.byte_size,
        'max-size-time': 0,
    }
    add_queues_to_element_group(
        element_group=pipeline_cfg,
        queue_properties=queue_properties,
        last_is_queue=False,
        next_should_be_queue=False,
        first_element=True,
    )


def add_queues_to_element_group(
    element_group: Union[Pipeline, ElementGroup],
    queue_properties: Dict[str, Any],
    last_is_queue: bool,
    next_should_be_queue: bool,
    first_element: bool,
):
    """Add queues to the pipeline or an element group before and after pyfunc elements."""

    elements = []
    for i, element in enumerate(element_group.elements):
        if isinstance(element, ElementGroup):
            if not element.init_condition.is_enabled:
                continue

            last_is_queue, next_should_be_queue = add_queues_to_element_group(
                element_group=element,
                queue_properties=queue_properties,
                last_is_queue=last_is_queue,
                next_should_be_queue=next_should_be_queue,
                first_element=first_element,
            )
            first_element = False
            elements.append(element)
            continue

        if (
            (next_should_be_queue and element.element != 'queue')
            or (isinstance(element, PyFuncElement) and not last_is_queue)
        ) and not first_element:
            elements.append(PipelineElement('queue', properties=queue_properties))

        elements.append(element)
        last_is_queue = element.element == 'queue'
        next_should_be_queue = isinstance(element, PyFuncElement)
        first_element = False

    element_group.elements = elements

    return last_is_queue, next_should_be_queue


def get_pipeline_element_stages(
    element_group: Union[Pipeline, ElementGroup],
    stage_idx_cache: Dict[str, int] = None,
) -> List[Union[str, List[str]]]:
    """Get the names of the pipeline or element group stages."""

    if stage_idx_cache is None:
        stage_idx_cache = {}
    stages = []
    for element in element_group.elements:
        if isinstance(element, ElementGroup):
            stages.append(get_pipeline_element_stages(element, stage_idx_cache))
        else:
            if isinstance(element, PyFuncElement):
                stage = f'pyfunc/{element.module}.{element.class_name}'
            elif element.name:
                stage = element.name
            else:
                stage = element.element
            if stage not in stage_idx_cache:
                stage_idx_cache[stage] = 0
            else:
                stage_idx_cache[stage] += 1
                stage = f'{stage}_{stage_idx_cache[stage]}'
            stages.append(stage)

    return stages


def build_pipeline_stages(element_stages: List[Union[str, List[str]]]):
    pipeline_stages = [
        (
            'source',
            VideoPipelineStagePayloadType.Frame,
            StageFunction.none(),
            StageFunction.none(),
        ),
        (
            'source-demuxer',
            VideoPipelineStagePayloadType.Frame,
            StageFunction.none(),
            StageFunction.none(),
        ),
        (
            'decode',
            VideoPipelineStagePayloadType.Frame,
            StageFunction.none(),
            StageFunction.none(),
        ),
        (
            'source-convert',
            VideoPipelineStagePayloadType.Frame,
            StageFunction.none(),
            StageFunction.none(),
        ),
        (
            'source-capsfilter',
            VideoPipelineStagePayloadType.Frame,
            StageFunction.none(),
            StageFunction.none(),
        ),
        (
            'muxer',
            VideoPipelineStagePayloadType.Frame,
            StageFunction.none(),
            StageFunction.none(),
        ),
        (
            'prepare-input',
            VideoPipelineStagePayloadType.Batch,
            StageFunction.none(),
            StageFunction.none(),
        ),
    ]
    for stage in element_stages:
        if isinstance(stage, str):
            pipeline_stages.append(
                (
                    stage,
                    VideoPipelineStagePayloadType.Batch,
                    StageFunction.none(),
                    StageFunction.none(),
                )
            )
        else:
            for x in stage:
                pipeline_stages.append(
                    (
                        x,
                        VideoPipelineStagePayloadType.Batch,
                        StageFunction.none(),
                        StageFunction.none(),
                    )
                )
    pipeline_stages.extend(
        [
            (
                'update-frame-meta',
                VideoPipelineStagePayloadType.Batch,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'demuxer',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'output-queue',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'frame-tag-filter',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'queue-tagged',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'sink-convert',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'encode',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'parse',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'sink-capsfilter',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'queue-not-tagged',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'frame-tag-funnel',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
            (
                'sink',
                VideoPipelineStagePayloadType.Frame,
                StageFunction.none(),
                StageFunction.none(),
            ),
        ]
    )

    return pipeline_stages
