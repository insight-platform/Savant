from typing import Any, Dict, Union

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
