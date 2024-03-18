"""Convert deepstream object meta to output format."""
import logging

import pyds
from savant_rs.primitives import Attribute, VideoObject, VideoFrame
from savant_rs.primitives.geometry import BBox, RBBox
from savant_rs.utils.symbol_mapper import parse_compound_key

from savant.api.builder import build_attribute_value
from savant.config.schema import FrameParameters
from savant.deepstream.utils.object import (
    nvds_get_obj_bbox,
    nvds_get_obj_uid,
    nvds_is_empty_object_meta,
)
from savant.meta.attribute import AttributeMeta
from savant.meta.constants import PRIMARY_OBJECT_KEY, UNTRACKED_OBJECT_ID
from savant.utils.logging import get_logger

logger = get_logger(__name__)


def nvds_obj_bbox_output_converter(
    nvds_obj_meta: pyds.NvDsObjectMeta, frame_params: FrameParameters
) -> RBBox:
    bbox = nvds_get_obj_bbox(nvds_obj_meta)
    if logger.isEnabledFor(logging.TRACE):
        logger.trace('Object DS bbox %s', bbox)
    if frame_params.padding and not frame_params.padding.keep:
        bbox.xc -= frame_params.padding.left
        bbox.yc -= frame_params.padding.top
        if logger.isEnabledFor(logging.TRACE):
            logger.trace(
                'Applied frame padding %s, bbox: %s', frame_params.padding, bbox
            )
    if isinstance(bbox, BBox):
        bbox = bbox.as_rbbox()
        bbox.angle = 0
    if logger.isEnabledFor(logging.TRACE):
        logger.trace('Object corrected bbox %s', bbox)

    return bbox


def nvds_obj_meta_output_converter(
    nvds_frame_meta: pyds.NvDsFrameMeta,
    nvds_obj_meta: pyds.NvDsObjectMeta,
    frame_params: FrameParameters,
    video_frame: VideoFrame,
) -> VideoObject:
    """Convert object meta to savant-rs format.

    :param nvds_frame_meta: NvDsFrameMeta
    :param nvds_obj_meta: NvDsObjectMeta
    :param frame_params: Frame parameters (width/height, to scale to [0..1])
    :return: Object meta in savant-rs format and its parent id.
    """
    model_name, label = parse_compound_key(nvds_obj_meta.obj_label)

    if logger.isEnabledFor(logging.TRACE):
        logger.trace(
            'Converting object meta for model "%s", label "%s".',
            model_name,
            label,
        )

    confidence = nvds_obj_meta.confidence
    if 0.0 < nvds_obj_meta.tracker_confidence < 1.0:  # specified confidence
        confidence = nvds_obj_meta.tracker_confidence

    bbox = nvds_obj_bbox_output_converter(nvds_obj_meta, frame_params)

    if nvds_obj_meta.object_id == UNTRACKED_OBJECT_ID:
        track_id = None
        track_box = None
    else:
        track_id = nvds_obj_meta.object_id
        track_box = bbox

    parent_id = None
    if (
        not nvds_is_empty_object_meta(nvds_obj_meta.parent)
        and nvds_obj_meta.parent.obj_label != PRIMARY_OBJECT_KEY
    ):
        parent_model_name, parent_label = parse_compound_key(
            nvds_obj_meta.parent.obj_label
        )
        if parent_model_name:
            parent_id = nvds_get_obj_uid(nvds_frame_meta, nvds_obj_meta.parent)

    video_object = video_frame.create_object(
        namespace=model_name,
        label=label,
        parent_id=parent_id,
        confidence=confidence,
        detection_box=bbox,
        track_id=track_id,
        track_box=track_box,
        attributes=[],
    )
    return video_object


def nvds_attr_meta_output_converter(
    attr_meta: AttributeMeta,
    is_persistent: bool = True,
) -> Attribute:
    """Convert attribute meta to savant-rs format.

    :param attr_meta: Attribute meta.
    :param is_persistent: Whether attribute is persistent.
    :return: Attribute meta in savant-rs format.
    """
    value = build_attribute_value(attr_meta.value, attr_meta.confidence)
    return Attribute(
        namespace=attr_meta.element_name,
        name=attr_meta.name,
        values=[value],
        is_persistent=is_persistent,
    )
