from typing import Union

from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    AttributeValueType,
    VideoFrame,
    VideoObject,
)
from savant_rs.primitives.geometry import BBox, RBBox
from savant_rs.video_object_query import MatchQuery

from savant.meta.constants import UNTRACKED_OBJECT_ID


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
        objects[obj.id] = parse_video_object(obj)

    for obj_id, parent in parents.items():
        child = objects[obj_id]
        child['parent_model_name'] = parent.namespace
        child['parent_label'] = parent.label
        child['parent_object_id'] = parent.get_track_id()

    return list(objects.values())


def parse_video_object(obj: VideoObject):
    track_id = obj.get_track_id()
    if track_id is None:
        track_id = UNTRACKED_OBJECT_ID

    return {
        'model_name': obj.namespace,
        'label': obj.label,
        'object_id': track_id,
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
