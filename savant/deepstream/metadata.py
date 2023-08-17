"""Convert deepstream object meta to output format."""
import logging
from typing import Any, Dict

import pyds
from savant_rs.primitives.geometry import RBBox
from savant_rs.utils.symbol_mapper import parse_compound_key

from savant.config.schema import FrameParameters
from savant.deepstream.utils import nvds_get_obj_bbox
from savant.deepstream.utils.object import nvds_is_empty_object_meta
from savant.meta.attribute import AttributeMeta
from savant.meta.constants import PRIMARY_OBJECT_KEY

logger = logging.getLogger(__name__)


def nvds_obj_meta_output_converter(
    nvds_obj_meta: pyds.NvDsObjectMeta,
    frame_params: FrameParameters,
) -> Dict[str, Any]:
    """Convert object meta to output format.

    :param nvds_obj_meta: NvDsObjectMeta
    :param frame_params: Frame parameters (width/height, to scale to [0..1]
    :return: resolution of output frame
    """
    model_name, label = parse_compound_key(nvds_obj_meta.obj_label)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            'Converting object meta for model "%s", label "%s".',
            model_name,
            label,
        )

    confidence = nvds_obj_meta.confidence
    if 0.0 < nvds_obj_meta.tracker_confidence < 1.0:  # specified confidence
        confidence = nvds_obj_meta.tracker_confidence

    bbox = nvds_get_obj_bbox(nvds_obj_meta)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Object DS bbox %s', bbox)
    if frame_params.padding and not frame_params.padding.keep:
        bbox.xc -= frame_params.padding.left
        bbox.yc -= frame_params.padding.top
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                'Applied frame padding %s, bbox: %s', frame_params.padding, bbox
            )
    bbox = dict(
        xc=bbox.xc,
        yc=bbox.yc,
        width=bbox.width,
        height=bbox.height,
        angle=bbox.angle if isinstance(bbox, RBBox) else 0.0,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('Object corrected bbox %s', bbox)
    # parse parent object
    parent_model_name, parent_label, parent_object_id = None, None, None
    if (
        not nvds_is_empty_object_meta(nvds_obj_meta.parent)
        and nvds_obj_meta.parent.obj_label != PRIMARY_OBJECT_KEY
    ):
        parent_model_name, parent_label = parse_compound_key(
            nvds_obj_meta.parent.obj_label
        )
        if parent_model_name:
            parent_object_id = nvds_obj_meta.parent.object_id

    return dict(
        model_name=model_name,
        label=label,
        object_id=nvds_obj_meta.object_id,
        bbox=bbox,
        confidence=confidence,
        attributes=[],
        parent_model_name=parent_model_name,
        parent_label=parent_label,
        parent_object_id=parent_object_id,
    )


def nvds_attr_meta_output_converter(attr_meta: AttributeMeta) -> Dict[str, Any]:
    """Convert attribute meta to output format.

    :param attr_meta: dict
    :return: dict
    """
    return attr_meta.__dict__
