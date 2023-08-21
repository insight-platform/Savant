from typing import Any, Dict, List, Union

from savant_rs.pipeline2 import VideoPipelineStagePayloadType

from savant.config.schema import (
    BufferQueuesParameters,
    ElementGroup,
    Pipeline,
    PipelineElement,
    PyFuncElement,
)


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
                stage_idx_cache[stage] = 1
            else:
                stage = f'{stage}_{stage_idx_cache[stage]}'
                stage_idx_cache[stage] += 1
            stages.append(stage)

    return stages


def build_pipeline_stages(element_stages: List[Union[str, List[str]]]):
    pipeline_stages = [
        ('source', VideoPipelineStagePayloadType.Frame),
        ('decode', VideoPipelineStagePayloadType.Frame),
        ('source-convert', VideoPipelineStagePayloadType.Frame),
        ('source-capsfilter', VideoPipelineStagePayloadType.Frame),
        ('muxer', VideoPipelineStagePayloadType.Frame),
        ('prepare-input', VideoPipelineStagePayloadType.Batch),
    ]
    for stage in element_stages:
        if isinstance(stage, str):
            pipeline_stages.append((stage, VideoPipelineStagePayloadType.Batch))
        else:
            for x in stage:
                pipeline_stages.append((x, VideoPipelineStagePayloadType.Batch))
    pipeline_stages.extend(
        [
            ('update-frame-meta', VideoPipelineStagePayloadType.Batch),
            ('demuxer', VideoPipelineStagePayloadType.Frame),
            ('output-queue', VideoPipelineStagePayloadType.Frame),
            ('frame-tag-filter', VideoPipelineStagePayloadType.Frame),
            ('queue-tagged', VideoPipelineStagePayloadType.Frame),
            ('sink-convert', VideoPipelineStagePayloadType.Frame),
            ('encode', VideoPipelineStagePayloadType.Frame),
            ('parse', VideoPipelineStagePayloadType.Frame),
            ('sink-capsfilter', VideoPipelineStagePayloadType.Frame),
            ('queue-not-tagged', VideoPipelineStagePayloadType.Frame),
            ('frame-tag-funnel', VideoPipelineStagePayloadType.Frame),
            ('sink', VideoPipelineStagePayloadType.Frame),
        ]
    )

    return pipeline_stages
