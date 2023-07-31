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

_attribute_value_to_python = {
    AttributeValueType.BBox: lambda x: x.as_bbox(),
    AttributeValueType.BBoxList: lambda x: x.as_bboxes(),
    AttributeValueType.Boolean: lambda x: x.as_boolean(),
    AttributeValueType.BooleanList: lambda x: x.as_booleans(),
    AttributeValueType.Bytes: lambda x: x.as_bytes(),
    AttributeValueType.Float: lambda x: x.as_float(),
    AttributeValueType.FloatList: lambda x: x.as_floats(),
    AttributeValueType.Integer: lambda x: x.as_integer(),
    AttributeValueType.IntegerList: lambda x: x.as_integers(),
    AttributeValueType.Intersection: lambda x: x.as_intersection(),
    AttributeValueType.Point: lambda x: x.as_point(),
    AttributeValueType.PointList: lambda x: x.as_points(),
    AttributeValueType.Polygon: lambda x: x.as_polygon(),
    AttributeValueType.PolygonList: lambda x: x.as_polygons(),
    AttributeValueType.String: lambda x: x.as_string(),
    AttributeValueType.StringList: lambda x: x.as_strings(),
}


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
    try:
        return _attribute_value_to_python[value.value_type](value)
    except KeyError:
        raise ValueError(f'Unknown attribute value type: {value.value_type}')


def parse_bbox(bbox: Union[BBox, RBBox]):
    return {
        'xc': bbox.xc,
        'yc': bbox.yc,
        'width': bbox.width,
        'height': bbox.height,
        'angle': bbox.angle if isinstance(bbox, RBBox) else 0,
    }
