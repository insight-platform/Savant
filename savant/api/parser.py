from typing import Tuple, Union

from savant_rs.primitives import (
    Attribute,
    AttributeValue,
    AttributeValueType,
    VideoFrame,
    VideoFrameTransformation,
    VideoObject,
)
from savant_rs.primitives.geometry import BBox, RBBox
from savant_rs.video_object_query import MatchQuery

from savant.api.constants import DEFAULT_TIME_BASE
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
    parsed = {
        'source_id': frame.source_id,
        'framerate': frame.framerate,
        'width': frame.width,
        'height': frame.height,
        'pts': convert_ts(frame.pts, frame.time_base),
        'keyframe': frame.keyframe,
        'codec': frame.codec,
        'dts': (
            convert_ts(frame.dts, frame.time_base) if frame.dts is not None else None
        ),
        'duration': (
            convert_ts(frame.duration, frame.time_base)
            if frame.duration is not None
            else None
        ),
        'metadata': {'objects': parse_video_objects(frame)},
        'tags': parse_tags(frame),
    }
    if frame.transformations:
        parsed['transformations'] = [
            parse_transformation(x) for x in frame.transformations
        ]

    return parsed


def convert_ts(ts: int, time_base: Tuple[int, int]):
    if time_base == DEFAULT_TIME_BASE:
        return ts
    tb_num, tb_denum = time_base
    target_num, target_denum = DEFAULT_TIME_BASE
    return ts * target_num * tb_denum // (target_denum * tb_num)


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


def parse_transformation(transformation: VideoFrameTransformation):
    if transformation.is_initial_size:
        width, height = transformation.as_initial_size
        return {'type': 'initial_size', 'width': width, 'height': height}
    if transformation.is_padding:
        left, top, right, bottom = transformation.as_padding
        return {
            'type': 'padding',
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom,
        }
    if transformation.is_resulting_size:
        width, height = transformation.as_resulting_size
        return {'type': 'resulting_size', 'width': width, 'height': height}
    if transformation.is_scale:
        width, height = transformation.as_scale
        return {'type': 'scale', 'width': width, 'height': height}
    raise ValueError(f'Unknown transformation type: {transformation}')
