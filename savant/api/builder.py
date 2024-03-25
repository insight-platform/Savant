from typing import Any, Dict, List, Optional, Tuple, Union

from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    IdCollisionResolutionPolicy,
    VideoFrame,
    VideoFrameContent,
    VideoObject,
)
from savant_rs.primitives.geometry import RBBox

from savant.api.constants import DEFAULT_NAMESPACE, DEFAULT_TIME_BASE
from savant.meta.constants import UNTRACKED_OBJECT_ID

_python_to_attribute_value = {
    bool: AttributeValue.boolean,
    int: AttributeValue.integer,
    float: AttributeValue.float,
    str: AttributeValue.string,
}
_python_to_attribute_value_list = {
    bool: AttributeValue.booleans,
    int: AttributeValue.integers,
    float: AttributeValue.floats,
    str: AttributeValue.strings,
}


def build_video_frame(
    source_id: str,
    framerate: str,
    width: int,
    height: int,
    pts: int,
    keyframe: bool,
    content: Optional[Union[bytes, Tuple[str, Optional[str]]]],
    codec: Optional[str] = None,
    dts: Optional[int] = None,
    duration: Optional[int] = None,
    objects: Optional[List[Dict[str, Any]]] = None,
    tags: Optional[Dict[str, Union[bool, int, float, str]]] = None,
    time_base: Tuple[int, int] = DEFAULT_TIME_BASE,
) -> VideoFrame:
    if content is None:
        content = VideoFrameContent.none()
    elif isinstance(content, bytes):
        content = VideoFrameContent.internal(content)
    else:
        content = VideoFrameContent.external(*content)
    frame = VideoFrame(
        source_id=source_id,
        framerate=framerate,
        width=width,
        height=height,
        codec=codec,
        content=content,
        keyframe=keyframe,
        pts=pts,
        dts=dts,
        duration=duration,
        time_base=time_base,
    )
    if objects:
        add_objects_to_video_frame(frame, objects)
    if tags:
        add_tags_to_video_frame(frame, tags)

    return frame


def add_objects_to_video_frame(
    frame: VideoFrame,
    objects: List[Dict[str, Any]],
):
    parents = {}
    children = {}
    for obj in objects:
        video_obj = add_video_object(frame, obj)
        children[video_obj.id] = obj
        track_id = video_obj.track_id
        if track_id is not None:
            parents[(video_obj.namespace, video_obj.label, track_id)] = video_obj.id

    for child_obj_id, child_obj in children.items():
        parent_id = child_obj.get('parent_object_id')
        parent_model_name = child_obj.get('parent_model_name')
        parent_label = child_obj.get('parent_label')
        if parent_id is None or parent_model_name is None or parent_label is None:
            continue
        frame.set_parent_by_id(
            child_obj_id,
            parents[(parent_model_name, parent_label, parent_id)],
        )


def add_video_object(frame, obj: Dict[str, Any]):
    attributes = obj.get('attributes')
    if attributes is not None:
        attributes = build_object_attributes(attributes)
    else:
        attributes = []
    bbox = build_bbox(obj['bbox'])
    track_id = obj['object_id']
    if track_id == UNTRACKED_OBJECT_ID:
        track_id = None
        track_box = None
    else:
        track_box = bbox

    return frame.create_object(
        namespace=obj['model_name'],
        label=obj['label'],
        confidence=obj['confidence'],
        detection_box=bbox,
        track_id=track_id,
        track_box=track_box,
        attributes=attributes,
    )


def build_bbox(bbox: Dict[str, Any]):
    return RBBox(
        xc=bbox['xc'],
        yc=bbox['yc'],
        width=bbox['width'],
        height=bbox['height'],
        angle=bbox.get('angle'),
    )


def build_object_attributes(attributes: List[Dict[str, Any]]):
    return [
        Attribute(
            namespace=attr['element_name'],
            name=attr['name'],
            values=[
                build_attribute_value(attr['value'], attr.get('confidence')),
            ],
        )
        for attr in attributes
    ]


def build_attribute_value(value: Any, confidence: Optional[float] = None):
    if isinstance(value, list):
        item_type = type(value[0]) if value else float
        try:
            return _python_to_attribute_value_list[item_type](value, confidence)
        except KeyError:
            raise ValueError(f'Unknown attribute value type: List[{item_type}]')
    else:
        try:
            return _python_to_attribute_value[type(value)](value, confidence)
        except KeyError:
            raise ValueError(f'Unknown attribute value type: {type(value)}')


def add_tags_to_video_frame(
    frame: VideoFrame,
    tags: Dict[str, Union[bool, int, float, str]],
):
    for name, value in tags.items():
        frame.set_persistent_attribute(
            namespace=DEFAULT_NAMESPACE,
            name=name,
            values=[build_attribute_value(value)],
        )
