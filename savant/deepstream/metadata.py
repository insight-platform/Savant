"""Convert deepstream object meta to output format."""
from typing import Any, Dict
import numpy as np
import pyds

from savant.config.schema import FrameParameters
from savant.converter.scale import scale_rbbox
from savant.deepstream.utils import nvds_get_rbbox
from savant.meta.attribute import AttributeMeta
from savant.utils.model_registry import ModelObjectRegistry


def nvds_obj_meta_output_converter(
    nvds_obj_meta: pyds.NvDsObjectMeta,
    frame_params: FrameParameters,
) -> Dict[str, Any]:
    """Convert object meta to output format.

    :param nvds_obj_meta: NvDsObjectMeta
    :param frame_params: Frame parameters (width/height, to scale to [0..1]
    :return: dict
    """
    model_name, label = ModelObjectRegistry.parse_model_object_key(
        nvds_obj_meta.obj_label
    )
    rect_params = nvds_obj_meta.detector_bbox_info.org_bbox_coords
    confidence = nvds_obj_meta.confidence
    if nvds_obj_meta.tracker_bbox_info.org_bbox_coords.width > 0:
        rect_params = nvds_obj_meta.tracker_bbox_info.org_bbox_coords
        if nvds_obj_meta.tracker_confidence < 1.0:  # specified confidence
            confidence = nvds_obj_meta.tracker_confidence

    if frame_params.padding and frame_params.padding.keep:
        frame_width = frame_params.total_width
        frame_height = frame_params.total_height
    else:
        frame_width = frame_params.width
        frame_height = frame_params.height

    # scale bbox to [0..1]
    # TODO: use a function to check bbox type explicitly
    if rect_params.width == 0:
        rbbox = nvds_get_rbbox(nvds_obj_meta)
        scaled_bbox = scale_rbbox(
            bboxes=np.array(
                [
                    [
                        rbbox.x_center,
                        rbbox.y_center,
                        rbbox.width,
                        rbbox.height,
                        rbbox.angle,
                    ]
                ]
            ),
            scale_factor_x=1 / frame_width,
            scale_factor_y=1 / frame_height,
        )[0]
        bbox = dict(
            xc=scaled_bbox[0],
            yc=scaled_bbox[1],
            width=scaled_bbox[2],
            height=scaled_bbox[3],
            angle=scaled_bbox[4],
        )
    else:
        obj_width = rect_params.width / frame_width
        obj_height = rect_params.height / frame_height
        bbox = dict(
            xc=rect_params.left / frame_width + obj_width / 2,
            yc=rect_params.top / frame_height + obj_height / 2,
            width=obj_width,
            height=obj_height,
        )
    if frame_params.padding and not frame_params.padding.keep:
        bbox['xc'] -= frame_params.padding.left / frame_width
        bbox['yc'] -= frame_params.padding.top / frame_height

    # parse parent object
    parent_model_name, parent_label, parent_object_id = None, None, None
    if nvds_obj_meta.parent and nvds_obj_meta.parent.obj_label != 'frame':
        parent_model_name, parent_label = ModelObjectRegistry.parse_model_object_key(
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
