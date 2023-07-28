from typing import Any, Dict, List, Optional, Tuple, Union

from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    IdCollisionResolutionPolicy,
    VideoFrame,
    VideoFrameContent,
    VideoObject,
)
from savant_rs.primitives import AttributeValueType
from savant_rs.primitives.geometry import BBox, RBBox
from savant_rs.video_object_query import IntExpression, MatchQuery

DEFAULT_NAMESPACE = 'default'
DEFAULT_TIME_BASE = (1, 10**9)  # nanosecond


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
    obj_dict = {}
    for obj in objects:
        obj = build_object(obj)
        frame.add_object(obj, IdCollisionResolutionPolicy.Error)
        obj_dict[obj.id] = obj

    for obj in objects:
        parent_id = obj.get('parent_object_id')
        if parent_id is None:
            continue
        frame.set_parent(
            MatchQuery.id(IntExpression.eq(obj['object_id'])),
            obj_dict[parent_id],
        )


def build_object(obj: Dict[str, Any]):
    attributes = obj.get('attributes')
    if attributes is not None:
        attributes = build_object_attributes(attributes)
    else:
        attributes = {}
    return VideoObject(
        id=obj['object_id'],
        namespace=obj['model_name'],
        label=obj['label'],
        detection_box=build_bbox(obj['bbox']),
        attributes=attributes,
        confidence=obj['confidence'],
    )


def build_bbox(bbox: Dict[str, Any]):
    angle = bbox.get('angle')
    if angle is None:
        return BBox(
            x=bbox['x'],
            y=bbox['y'],
            width=bbox['width'],
            height=bbox['height'],
        )
    return RBBox(
        xc=bbox['xc'],
        yc=bbox['yc'],
        width=bbox['width'],
        height=bbox['height'],
        angle=angle,
    )


def build_object_attributes(attributes: List[Dict[str, Any]]):
    return {
        (attr['element_name'], attr['name']): Attribute(
            namespace=attr['element_name'],
            name=attr['name'],
            values=[
                build_attribute_value(attr['value'], attr.get('confidence')),
            ],
        )
        for attr in attributes
    }


def build_attribute_value(
    value: Union[bool, int, float, str, List[float]],
    confidence: Optional[float] = None,
):
    if isinstance(value, bool):
        return AttributeValue.boolean(value, confidence=confidence)
    elif isinstance(value, int):
        return AttributeValue.integer(value, confidence=confidence)
    elif isinstance(value, float):
        return AttributeValue.float(value, confidence=confidence)
    elif isinstance(value, str):
        return AttributeValue.string(value, confidence=confidence)
    elif isinstance(value, list):
        return AttributeValue.floats(value, confidence=confidence)


def add_tags_to_video_frame(
    frame: VideoFrame,
    tags: Dict[str, Union[bool, int, float, str]],
):
    for name, value in tags.items():
        frame.set_attribute(
            Attribute(
                namespace=DEFAULT_NAMESPACE,
                name=name,
                values=[build_attribute_value(value)],
            )
        )


def parse_video_frame(frame: VideoFrame):
    # TODO: add content to metadata if its not embedded
    return {
        'source_id': frame.source_id,
        'framerate': frame.framerate,
        'width': frame.width,
        'height': frame.height,
        'pts': frame.pts,
        'keyframe': frame.keyframe,
        'codec': frame.codec,
        'dts': frame.dts,
        'duration': frame.duration,
        'metadata': {'objects': parse_video_objects(frame)},
        'tags': parse_tags(frame),
    }


def parse_tags(frame: VideoFrame):
    return {
        name: parse_attribute_value(frame.get_attribute(namespace, name).values[0])
        for namespace, name in frame.attributes
    }


def parse_video_objects(frame: VideoFrame):
    parents = {}
    objects = {}
    for obj in frame.access_objects(MatchQuery.idle()):
        for child in frame.get_children(obj.id):
            parents[child.id] = obj
        objects[obj.id] = {
            'model_name': obj.namespace,
            'label': obj.label,
            'object_id': obj.id,
            'bbox': parse_bbox(obj.detection_box),
            'confidence': obj.confidence,
            'attributes': [
                parse_attribute(obj.get_attribute(namespace, name))
                for namespace, name in obj.attributes
            ],
            'parent_model_name': None,
            'parent_label': None,
            'parent_object_id': None,
        }

    for obj_id, parent in parents.items():
        child = objects[obj_id]
        child['parent_model_name'] = parent.namespace
        child['parent_label'] = parent.label
        child['parent_object_id'] = parent.id

    return list(objects.values())


def parse_attribute(attribute: Attribute):
    value = attribute.values[0]
    return {
        'element_name': attribute.namespace,
        'name': attribute.name,
        'value': parse_attribute_value(value),
        'confidence': value.confidence,
    }


def parse_attribute_value(value: AttributeValue):
    # primitive
    if value.value_type == AttributeValueType.Boolean:
        return value.as_boolean()
    if value.value_type == AttributeValueType.Integer:
        return value.as_integer()
    if value.value_type == AttributeValueType.Float:
        return value.as_float()
    if value.value_type == AttributeValueType.String:
        return value.as_string()
    if value.value_type == AttributeValueType.Bytes:
        return value.as_bytes()

    # list of primitives
    if value.value_type == AttributeValueType.BooleanList:
        return value.as_booleans()
    if value.value_type == AttributeValueType.IntegerList:
        return value.as_integers()
    if value.value_type == AttributeValueType.FloatList:
        return value.as_floats()
    if value.value_type == AttributeValueType.StringList:
        return value.as_strings()

    # object
    if value.value_type == AttributeValueType.BBox:
        return value.as_bbox()
    if value.value_type == AttributeValueType.Point:
        return value.as_point()
    if value.value_type == AttributeValueType.Polygon:
        return value.as_polygon()
    if value.value_type == AttributeValueType.Intersection:
        return value.as_intersection()

    # list of objects
    if value.value_type == AttributeValueType.BBoxList:
        return value.as_bboxes()
    if value.value_type == AttributeValueType.PointList:
        return value.as_points()
    if value.value_type == AttributeValueType.PolygonList:
        return value.as_polygons()

    raise ValueError(f'Unknown attribute value type: {value.value_type}')


def parse_bbox(bbox: Union[BBox, RBBox]):
    return {
        'xc': bbox.xc,
        'yc': bbox.yc,
        'width': bbox.width,
        'height': bbox.height,
        'angle': bbox.angle if isinstance(bbox, RBBox) else 0,
    }
